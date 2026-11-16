"""Browser chat with the WIiTek agent.

Run with ``uv run streamlit run app.py`` and open http://localhost:8501. Needs the database running and OPENAI_API_KEY
set, like ``main.py``.

Laid out like common chat apps: the sidebar holds "new chat", a search over this tab's conversations, the list of them
and the interface language (PL/EN); an empty conversation shows a welcome screen with the question box and example
questions. The interface language changes only the page's own labels and examples — the agent answers in the language
of the question either way. The look (dark theme, fonts, cards) comes from ``.streamlit/config.toml`` and
``ui/style.css``.

Streamlit reruns this script from the top on every interaction, so what the page shows lives in ``st.session_state``;
what the agent remembers lives separately, in its checkpointer, under each conversation's ``thread_id``.
"""

from __future__ import annotations

import base64
import html
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

import streamlit as st

_ROOT = Path(__file__).resolve().parent

# The application code lives in src/ and imports itself by module name
# (``from config import ...``), so src/ has to be on the import path.
sys.path.insert(0, str(_ROOT / "src"))

from langchain_core.messages import BaseMessage
from langgraph.graph.state import CompiledStateGraph

from agent import ask, build_agent
from config import AgentSettings

STYLE_PATH = _ROOT / "ui" / "style.css"
# The "[W]" logo: sidebar brand, the assistant's avatar and the browser tab icon.
LOGO_PATH = _ROOT / "ui" / "logo.png"

# The page's own labels, per interface language.
UI_TEXT = {
    "PL": {
        "subtitle": "Asystent studenta WIiT PP",
        "new_chat": "Nowa rozmowa",
        "search": "Szukaj w rozmowach",
        "today": "Dzisiaj",
        "last_week": "Ostatnie 7 dni",
        "older": "Starsze",
        "language": "Język",
        "data_note": "Dane: progi punktowe (PDF), katalog ECTS, karty przedmiotów, Informator PP.",
        "badge": "Każda liczba w odpowiedzi ma podane źródło",
        "welcome": "W czym mogę pomóc?",
        "welcome_text": "Pytaj o progi punktowe, plany studiów, karty przedmiotów i konsultacje "
        "na Wydziale Informatyki i Telekomunikacji PP.",
        "input": "Zadaj pytanie o studia na WIiT…",
        "follow_up": "Zadaj pytanie uzupełniające…",
        "examples": "Zacznij od przykładu",
        "disclaimer": "WIiTek może się pomylić. Decyzje rekrutacyjne sprawdzaj w oficjalnych komunikatach uczelni.",
        "searching": "Szukam w danych",
        "searched": "Przeszukano dane: {topics}",
        "no_tools": "Odpowiedź bez sięgania do danych",
        "failed": "Nie udało się odpowiedzieć",
        "sources": "Źródła",
        "course_card": "Karta przedmiotu",
        "thresholds": "Progi punktowe (PDF)",
        "staff_card": "Informator PP",
        "ects_catalog": "Katalog ECTS",
        "web_page": "Strona",
        "data_of": "dane z {date}",
    },
    "EN": {
        "subtitle": "WIiT PUT student assistant",
        "new_chat": "New chat",
        "search": "Search chats",
        "today": "Today",
        "last_week": "Last 7 days",
        "older": "Older",
        "language": "Language",
        "data_note": "Data: admission thresholds (PDF), ECTS catalog, course cards, Informator PP.",
        "badge": "Every number in an answer cites its source",
        "welcome": "How can I help?",
        "welcome_text": "Ask about admission thresholds, study plans, course cards and office hours "
        "at the Faculty of Computing and Telecommunications, Poznań University of Technology.",
        "input": "Ask about studying at WIiT…",
        "follow_up": "Ask a follow-up question…",
        "examples": "Start from an example",
        "disclaimer": "WIiTek can make mistakes. Check admission decisions in the university's official announcements.",
        "searching": "Searching the data",
        "searched": "Searched: {topics}",
        "no_tools": "Answered without the data",
        "failed": "Could not answer",
        "sources": "Sources",
        "course_card": "Course card",
        "thresholds": "Admission thresholds (PDF)",
        "staff_card": "Informator PP",
        "ects_catalog": "ECTS catalog",
        "web_page": "Web page",
        "data_of": "data of {date}",
    },
}
LANGUAGES = list(UI_TEXT)

# What each tool searches, for the "Searched: …" line above an answer.
TOOL_TOPICS = {
    "get_admission_threshold": {"PL": "progi punktowe", "EN": "admission thresholds"},
    "get_course": {"PL": "plany studiów", "EN": "study plans"},
    "search_course_cards": {"PL": "karty przedmiotów", "EN": "course cards"},
    "get_staff": {"PL": "pracownicy", "EN": "staff"},
    "get_staff_function": {"PL": "pracownicy", "EN": "staff"},
    "get_office_hour": {"PL": "konsultacje", "EN": "office hours"},
    "get_major": {"PL": "kierunki", "EN": "majors"},
    "get_department": {"PL": "jednostki", "EN": "units"},
    "get_faculties": {"PL": "wydziały", "EN": "faculties"},
}

# Example cards on the welcome screen: (icon, topic, question); picking one asks it as if typed. Each question was
# tried on the agent; together they cover several majors and every kind of data.
EXAMPLES = {
    "PL": [
        (":material/show_chart:", "Progi punktowe", "Jaki był próg punktowy na Teleinformatykę w 2023 roku?"),
        (":material/calendar_month:", "Plany studiów", "Jakie przedmioty są na 3. semestrze Bioinformatyki?"),
        (
            ":material/description:",
            "Karty przedmiotów",
            "Jak się zalicza Technikę kwantową na Mikroelektronice i komunikacji cyfrowej?",
        ),
        (":material/person:", "Prowadzący i konsultacje", "Kto prowadzi Bazy danych na Informatyce?"),
    ],
    "EN": [
        (
            ":material/show_chart:",
            "Admission thresholds",
            "What was the admission threshold for Computer Science in 2024?",
        ),
        (":material/calendar_month:", "Study plans", "Which courses are in the 4th semester of Teleinformatics?"),
        (
            ":material/description:",
            "Course cards",
            "What do students learn in Virtualization Methods on Cybersecurity?",
        ),
        (":material/person:", "Lecturers and office hours", "Who teaches Databases on Computer Science?"),
    ],
}

# A course card link as the database stores it: spaces and brackets unescaped, e.g.
# "https://put.poznan.pl/cards/.../studia pierwszego stopnia/4/Systemy baz danych (SBD).pdf". Markdown would end the link
# at the first space, so these are escaped before an answer is shown.
COURSE_CARD_LINK = re.compile(r"https://put\.poznan\.pl/cards/[^\n<>]*?\.pdf")
# The answer's closing source line, which the agent writes as "Źródło: … (dane z 2026-11-15)" or in English.
SOURCE_LINE = re.compile(r"^\W*(źródł[oa]|sources?)\W*:", re.IGNORECASE | re.MULTILINE)
OTHER_LINK = re.compile(r"https?://[^\s()<>\[\]]+")
THRESHOLDS_FILE = re.compile(r"PDF:\s*([^()\n]+?\.pdf)", re.IGNORECASE)
DATA_DATE = re.compile(r"(?:dane z|data of|data from)\s+(\d{4}-\d{2}-\d{2})", re.IGNORECASE)


@st.cache_resource
def logo_data_uri() -> str:
    """Return the logo as a data URI, for the sidebar brand (HTML there cannot point at a local file).

    Returns:
        str: "data:image/png;base64,…".
    """
    return "data:image/png;base64," + base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")


@st.cache_resource
def get_agent() -> CompiledStateGraph:
    """Build the agent once per process and share it between all browser sessions.

    One agent means one in-memory checkpointer: conversations are kept apart by their ``thread_id``, and every
    conversation is lost when the process stops.

    Returns:
        CompiledStateGraph: The agent from ``build_agent``.
    """
    return build_agent(AgentSettings())


def tool_calls_of(messages: list[BaseMessage]) -> list[dict]:
    """List the tool calls of one turn.

    Args:
        messages: One turn's messages, from ``ask``.

    Returns:
        list[dict]: One ``{"name", "args"}`` per call in order, e.g. ``{"name": "get_course", "args": {...}}``.
    """
    return [
        {"name": call["name"], "args": call["args"]}
        for message in messages
        for call in getattr(message, "tool_calls", None) or []
    ]


def linkable(text: str) -> str:
    """Escape the spaces and brackets of course card links, so Markdown shows each as one whole link.

    Args:
        text: Answer text.

    Returns:
        str: The same text; only inside course card links (".pdf" under put.poznan.pl/cards/) are " ", "(" and ")"
            replaced by "%20", "%28" and "%29".
    """
    return COURSE_CARD_LINK.sub(
        lambda link: link.group().replace(" ", "%20").replace("(", "%28").replace(")", "%29"), text
    )


def describe_source(url: str, text: dict[str, str]) -> tuple[str, str]:
    """Name a cited link for its source card.

    Args:
        url: The link as cited.
        text: The interface language's labels (``UI_TEXT``).

    Returns:
        tuple[str, str]: Title and kind, e.g. ("Systemy baz danych (SBD)", "Karta przedmiotu · Informatyka").
    """
    if COURSE_CARD_LINK.fullmatch(url):
        parts = unquote(url).split("/")
        # .../cards/<academic year>/<major[_English name]>/.../<course (CODE)>.pdf
        major = parts[5].split("_")[0] if len(parts) > 5 else ""
        return parts[-1].removesuffix(".pdf"), f"{text['course_card']} · {major}".rstrip(" ·")
    if "informator.put.poznan.pl" in url:
        return text["staff_card"], url.split("//", 1)[-1].split("/", 1)[0]
    if "/karty-ects/" in url:
        return text["ects_catalog"], unquote(url).split("/karty-ects/", 1)[-1].split("/")[1]
    return text["web_page"], url.split("//", 1)[-1].split("/", 1)[0]


def split_sources(answer: str, text: dict[str, str]) -> tuple[str, list[dict]]:
    """Take the closing source line off an answer and turn what it cites into source cards.

    The agent ends an answer with a line like "Źródło: <link or PDF> (dane z 2026-11-15)". It is shown as cards under
    the answer instead; when nothing in it can be recognised, the answer is left whole.

    Args:
        answer: The agent's final message.
        text: The interface language's labels (``UI_TEXT``).

    Returns:
        tuple[str, list[dict]]: The answer without the source line, and one ``{"title", "detail", "url"}`` per cited
            source (``url`` None for the thresholds PDF, which is a local file).
    """
    matches = list(SOURCE_LINE.finditer(answer))
    if not matches:
        return answer, []
    body, block = answer[: matches[-1].start()].rstrip(), answer[matches[-1].start() :]
    date = DATA_DATE.search(block)
    when = text["data_of"].format(date=date.group(1)) if date else ""

    sources: list[dict] = []
    for pdf in THRESHOLDS_FILE.findall(block):
        sources.append({"title": text["thresholds"], "detail": pdf.strip().rsplit("/", 1)[-1], "url": None})
    card_links = COURSE_CARD_LINK.findall(block)
    rest = COURSE_CARD_LINK.sub(" ", block)
    for url in [*card_links, *(link.rstrip(".,;") for link in OTHER_LINK.findall(rest))]:
        if any(source["url"] == url for source in sources):
            continue
        title, kind = describe_source(url, text)
        sources.append({"title": title, "detail": kind, "url": url})
    if not sources:
        return answer, []
    for source in sources:
        source["detail"] = " · ".join(part for part in (source["detail"], when) if part)
    return body, sources


def show_sources(sources: list[dict], text: dict[str, str]) -> None:
    """Show the cited sources as numbered cards; a linked one opens in a new tab.

    Args:
        sources: From ``split_sources``.
        text: The interface language's labels (``UI_TEXT``).
    """
    cards = []
    for number, source in enumerate(sources, start=1):
        inner = (
            f'<span class="n">{number}</span><span><div class="t">{html.escape(source["title"])}</div>'
            f'<div class="d">{html.escape(source["detail"])}</div></span>'
        )
        if source["url"]:
            href = html.escape(linkable(source["url"]), quote=True)
            cards.append(f'<a class="wt-source" href="{href}" target="_blank" rel="noopener">{inner}</a>')
        else:
            cards.append(f'<div class="wt-source">{inner}</div>')
    st.html(f'<div class="wt-sources-label">{text["sources"]}</div><div class="wt-sources">{"".join(cards)}</div>')


def searched_label(tool_calls: list[dict], language: str) -> str:
    """Label of the folded tool calls above an answer: what kind of data the agent searched.

    Args:
        tool_calls: From ``tool_calls_of``.
        language: The interface language, "PL" or "EN".

    Returns:
        str: E.g. "Przeszukano dane: progi punktowe, kierunki", or the no-tools label.
    """
    text = UI_TEXT[language]
    if not tool_calls:
        return text["no_tools"]
    topics = dict.fromkeys(TOOL_TOPICS.get(call["name"], {}).get(language, call["name"]) for call in tool_calls)
    return text["searched"].format(topics=", ".join(topics))


def show_tool_calls(tool_calls: list[dict]) -> None:
    """Write the turn's tool calls, one code line each, into the container that is open.

    Args:
        tool_calls: From ``tool_calls_of``.
    """
    for call in tool_calls:
        st.code(f"{call['name']}({call['args']})", language=None, wrap_lines=True)


def show_answer_body(answer: str, language: str) -> None:
    """Show an answer's text and, under it, the source cards made from its source line.

    Args:
        answer: The agent's final message.
        language: The interface language, "PL" or "EN".
    """
    body, sources = split_sources(answer, UI_TEXT[language])
    st.markdown(linkable(body))
    if sources:
        show_sources(sources, UI_TEXT[language])


def show_question(question: str) -> None:
    """Show the user's question as a chat bubble.

    Args:
        question: The question as asked.
    """
    with st.chat_message("user"):
        st.markdown(question)


def show_turn(turn: dict, language: str) -> None:
    """Show a stored question and answer, with the answer's tool calls folded away above it.

    Args:
        turn: ``{"question", "answer", "tool_calls"}`` as stored in the conversation.
        language: The interface language, "PL" or "EN".
    """
    show_question(turn["question"])
    with st.chat_message("assistant", avatar=str(LOGO_PATH)):
        with st.expander(searched_label(turn["tool_calls"], language), icon=":material/check:", type="compact"):
            show_tool_calls(turn["tool_calls"])
        show_answer_body(turn["answer"], language)


def current_turns() -> list[dict]:
    """Return the open conversation's turns (the list itself, so appending to it stores a turn).

    Returns:
        list[dict]: ``{"question", "answer", "tool_calls"}`` per turn, oldest first.
    """
    return st.session_state.conversations[st.session_state.current]["turns"]


def answer_question(question: str, language: str) -> None:
    """Ask the agent within the open conversation, store the turn and show the answer.

    The turn is stored as soon as the agent has answered, before anything is drawn: a click elsewhere during the wait
    (another conversation, the language) reruns the page at the next drawing call, and the answer must survive it.

    Args:
        question: The question, typed or picked from the examples.
        language: The interface language, "PL" or "EN".
    """
    text = UI_TEXT[language]
    turns = current_turns()
    show_question(question)
    with st.chat_message("assistant", avatar=str(LOGO_PATH)):
        with st.status(f":shimmer[{text['searching']}]", type="compact") as status:
            try:
                messages = ask(get_agent(), question, st.session_state.current)
            # No database, no API key, an API error: whatever it is, the user gets a message, not a stack trace.
            except Exception as error:
                status.update(label=text["failed"], state="error")
                st.error(str(error), title=text["failed"])
                return
            answer, tool_calls = messages[-1].content, tool_calls_of(messages)
            turns.append({"question": question, "answer": answer, "tool_calls": tool_calls})
            show_tool_calls(tool_calls)
            status.update(label=searched_label(tool_calls, language), state="complete")
        show_answer_body(answer, language)


def start_new_conversation() -> None:
    """Switch to a new, empty conversation (a fresh ``thread_id``), unless the open one is still empty."""
    current = st.session_state.get("current")
    if current is not None and not st.session_state.conversations[current]["turns"]:
        return
    thread_id = str(uuid.uuid4())
    st.session_state.conversations[thread_id] = {"started": datetime.now(), "turns": []}
    st.session_state.current = thread_id


def open_conversation(thread_id: str) -> None:
    """Switch to an earlier conversation of this tab; the agent still remembers it under its ``thread_id``.

    Args:
        thread_id: The conversation to show.
    """
    st.session_state.current = thread_id


def ask_example(question: str) -> None:
    """Queue an example question, to be asked in this rerun as if typed.

    Args:
        question: The example card's question.
    """
    st.session_state.pending_question = question


def age_group(started: datetime, text: dict[str, str]) -> str:
    """Name the sidebar group a conversation falls in by when it started.

    Args:
        started: When the conversation was created.
        text: The interface language's labels (``UI_TEXT``).

    Returns:
        str: "Dzisiaj", "Ostatnie 7 dni" or "Starsze" (or the English labels).
    """
    days = (datetime.now().date() - started.date()).days
    return text["today"] if days == 0 else text["last_week"] if days < 7 else text["older"]


def show_sidebar(language: str) -> None:
    """Draw the sidebar: brand, new chat, conversation search, conversations by age, interface language.

    Drawn before the chat, so it stays usable while the agent works; a conversation's first turn reruns the page to list
    it.

    Args:
        language: The interface language, "PL" or "EN".
    """
    text = UI_TEXT[language]
    with st.sidebar:
        st.html(
            f'<div class="wt-brand"><img class="wt-logo" src="{logo_data_uri()}" alt="">'
            f'<div><div class="wt-name">WIiTek</div><div class="wt-sub">{text["subtitle"]}</div></div></div>'
        )
        st.button(
            text["new_chat"],
            icon=":material/add:",
            type="primary",
            shortcut="Ctrl+K",
            on_click=start_new_conversation,
            width="stretch",
        )
        query = st.text_input(
            text["search"], placeholder=text["search"], icon=":material/search:", label_visibility="collapsed"
        )
        shown_group = None
        # Newest first; only conversations with a question, matching the search.
        for thread_id, conversation in reversed(st.session_state.conversations.items()):
            turns = conversation["turns"]
            if not turns or query.lower() not in " ".join(turn["question"] for turn in turns).lower():
                continue
            group = age_group(conversation["started"], text)
            if group != shown_group:
                st.html(f'<div class="wt-group">{group}</div>')
                shown_group = group
            st.button(
                turns[0]["question"],
                key=f"conversation_{thread_id}",
                on_click=open_conversation,
                args=(thread_id,),
                type="tertiary" if thread_id == st.session_state.current else "secondary",
                width="stretch",
            )
        st.divider()
        # One row that never wraps, so switching the language does not move the control to another line.
        with st.container(horizontal=True, vertical_alignment="center", wrap=False):
            st.markdown(text["language"])
            st.segmented_control(
                text["language"],
                LANGUAGES,
                default=LANGUAGES[0],
                required=True,
                key="ui_language",
                label_visibility="collapsed",
            )
        st.html(f'<div class="wt-sidebar-note">{text["data_note"]}</div>')


def show_welcome(language: str) -> str | None:
    """Show the empty conversation's welcome screen: badge, heading, the question box and example cards.

    Args:
        language: The interface language, "PL" or "EN".

    Returns:
        str | None: The question typed into the welcome screen's box, if one was just sent.
    """
    text = UI_TEXT[language]
    st.html(
        f'<div class="wt-hero"><span class="wt-badge"><b>✓</b> {text["badge"]}</span>'
        f"<h1>{text['welcome']}</h1><p>{text['welcome_text']}</p></div>"
    )
    # Inside a container the chat input sits here, under the heading, instead of at the bottom of the page.
    with st.container():
        question = st.chat_input(text["input"], key="welcome_input")
    st.html(f'<div class="wt-examples-label">{text["examples"]}</div>')
    columns = st.columns(2)
    for number, (icon, topic, example) in enumerate(EXAMPLES[language]):
        with columns[number % 2]:
            st.button(
                f"**{topic}**  \n{example}",
                key=f"ex_{language}_{number}",
                icon=icon,
                on_click=ask_example,
                args=(example,),
                width="stretch",
                wrap=True,
            )
    st.html(f'<div class="wt-disclaimer">{text["disclaimer"]}</div>')
    return question


def main() -> None:
    """Draw the sidebar and the chat page, answering the question just asked, if any."""
    st.set_page_config(page_title="WIiTek", page_icon=str(LOGO_PATH), layout="centered")
    st.html(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>")
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}  # thread_id -> {"started": datetime, "turns": [...]}
        start_new_conversation()
    # The language control is drawn in the sidebar below; its value from the previous run is already in the state.
    language = st.session_state.get("ui_language", LANGUAGES[0])
    show_sidebar(language)

    turns = current_turns()
    question = st.session_state.pop("pending_question", None)
    if turns:
        st.html(f'<div class="wt-chat-title">{html.escape(turns[0]["question"])}</div>')
        for turn in turns:
            show_turn(turn, language)
        question = st.chat_input(UI_TEXT[language]["follow_up"], submit_mode="disable") or question
    else:
        # In a placeholder, so it can be cleared at once when the conversation starts: otherwise the welcome screen
        # stays on the page, greyed out, until the answer is done.
        welcome = st.empty()
        with welcome.container():
            question = show_welcome(language) or question
        if question:
            welcome.empty()

    if question:
        first_turn = not turns
        answer_question(question, language)
        if first_turn and turns:
            st.rerun()  # list the new conversation in the sidebar, drawn before it had a title


if __name__ == "__main__":
    main()
