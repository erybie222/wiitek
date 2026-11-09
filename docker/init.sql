CREATE EXTENSION IF NOT EXISTS vector;
-- Lets name lookups ignore Polish diacritics ("Cyberbezpieczenstwo" finds "Cyberbezpieczeństwo").
CREATE EXTENSION IF NOT EXISTS unaccent;

CREATE TABLE IF NOT EXISTS faculties (
    symbol VARCHAR(10) CONSTRAINT pk_faculties PRIMARY KEY,
    name VARCHAR(50) CONSTRAINT uk_faculty_name UNIQUE
);

CREATE TABLE IF NOT EXISTS majors (
    id SERIAL CONSTRAINT pk_majors PRIMARY KEY,
    polish_name VARCHAR(100) NOT NULL,
    degree VARCHAR(50) NOT NULL CHECK (degree IN ('bachelor-of-engineering', 'masters')),
    study_mode VARCHAR(50) NOT NULL CHECK (study_mode IN ('full-time', 'part-time')),
    profile VARCHAR(50) NOT NULL DEFAULT 'general-academic' CHECK (profile IN ('general-academic', 'practical')),
    faculty VARCHAR(10) CONSTRAINT fk_major_faculty REFERENCES faculties(symbol),
    -- Official English name of an English-taught major; NULL for a Polish-taught
    -- one, so a non-NULL value is what marks a major as taught in English.
    english_name VARCHAR(100),
    -- The university runs some majors in both languages under the same Polish
    -- name ("architektura" and "architektura/Architecture"), so the English
    -- name is part of a major's identity. NULLS NOT DISTINCT: without it two
    -- Polish-taught rows (english_name NULL) would never conflict, and every
    -- re-ingestion would insert a duplicate instead of updating.
    CONSTRAINT uk_major UNIQUE NULLS NOT DISTINCT (polish_name, degree, study_mode, english_name)
);

CREATE TABLE IF NOT EXISTS admission_thresholds (
    id SERIAL CONSTRAINT pk_admission_thresholds PRIMARY KEY,
    major_id INT NOT NULL CONSTRAINT fk_threshold_major REFERENCES majors(id) ON DELETE CASCADE,
    year INT NOT NULL,
    round INT NOT NULL DEFAULT 1 CONSTRAINT ck_round CHECK(round BETWEEN 1 AND 2),
    min_points NUMERIC(5,2) NOT NULL,
    capacity INT NOT NULL,
    source_url TEXT,
    last_updated TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uk_admission_thresholds UNIQUE (major_id, year, round)
);

CREATE TABLE IF NOT EXISTS courses (
    course_code VARCHAR(50) NOT NULL,
    major_id INT NOT NULL CONSTRAINT fk_course_major REFERENCES majors(id) ON DELETE CASCADE,
    semester INT NOT NULL CHECK (semester BETWEEN 1 AND 8),
    name VARCHAR(200) NOT NULL,
    is_english BOOLEAN NOT NULL DEFAULT FALSE,
    is_obligatory BOOLEAN NOT NULL DEFAULT TRUE,
    ects INT NOT NULL CHECK (ects >= 0),
    hours_lecture INT NOT NULL DEFAULT 0,
    hours_classes INT NOT NULL DEFAULT 0,
    hours_lab INT NOT NULL DEFAULT 0,
    hours_project INT NOT NULL DEFAULT 0,
    assessment_type VARCHAR(50),
    source_url TEXT,
    last_updated TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    -- One row per catalog row: the same code (e.g. a PE elective) is listed on every
    -- major's page, and some electives in more than one semester.
    CONSTRAINT pk_courses PRIMARY KEY (course_code, major_id, semester)
);

CREATE TABLE IF NOT EXISTS departments (
    name VARCHAR(100) CONSTRAINT pk_departments PRIMARY KEY,
    symbol VARCHAR(10),
    parent VARCHAR(100) CONSTRAINT fk_department_parent REFERENCES departments(name),
    unit_type VARCHAR(50),
    description TEXT,
    faculty VARCHAR(10) NOT NULL CONSTRAINT fk_department_faculty REFERENCES faculties(symbol),
    address TEXT,
    phone VARCHAR(100),
    email VARCHAR(100),
    source_url TEXT,
    last_updated TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS staff (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    academic_title VARCHAR(100),
    position VARCHAR(100),
    user_type VARCHAR(50),
    department VARCHAR(100) CONSTRAINT fk_staff_department REFERENCES departments(name),
    email VARCHAR(100) CONSTRAINT uk_staff_email UNIQUE,
    phone VARCHAR(100),
    source_url TEXT,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS staff_functions (
    id SERIAL PRIMARY KEY,
    staff_id INT NOT NULL CONSTRAINT fk_staff_functions_staff REFERENCES staff(id) ON DELETE CASCADE,
    function_name VARCHAR(150) NOT NULL,
    department_name VARCHAR(100),
    source_url TEXT,
    last_updated TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    -- NULLS NOT DISTINCT: a function with no department_name is still a duplicate of itself.
    CONSTRAINT uk_staff_function UNIQUE NULLS NOT DISTINCT (staff_id, function_name, department_name)
);

CREATE TABLE IF NOT EXISTS office_hours (
    id SERIAL PRIMARY KEY,
    staff_id INT NOT NULL CONSTRAINT fk_office_hours_staff REFERENCES staff(id) ON DELETE CASCADE,
    day_of_week VARCHAR(20) NOT NULL,
    start_time TIME NOT NULL,
    end_time TIME NOT NULL,
    location VARCHAR(100),
    comment TEXT,
    source_url TEXT,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uk_office_hours UNIQUE (staff_id, day_of_week, start_time)
);

-- Course card text: one row per section of one course's card PDF.
-- Belongs to a course row (a PE syllabus listed on twelve majors is stored
-- twelve times), and goes when the course leaves the catalog: ingestion
-- updates courses in place, so the cascade fires only for a removed course.
CREATE TABLE IF NOT EXISTS course_card_sections (
    id SERIAL CONSTRAINT pk_course_card_sections PRIMARY KEY,
    course_code VARCHAR(50) NOT NULL,
    major_id INT NOT NULL,
    semester INT NOT NULL,
    -- The PDF this text was read from. Usually the course's current
    -- source_url; an older one when the course's new PDF could not be read,
    -- so a citation always names the file the text really comes from.
    source_url TEXT NOT NULL,
    -- The section heading as printed in the PDF, e.g. "Literatura"; the
    -- first page's two columns are stored together as "Koordynatorzy i wykładowcy".
    section VARCHAR(100) NOT NULL,
    -- Position of the piece within its section; 0 while sections are not split.
    chunk_index INT NOT NULL DEFAULT 0,
    content TEXT NOT NULL,
    -- The three below are set together by the embedding step and stay NULL
    -- until it runs (it is skipped without an OpenAI key). content_hash is
    -- the sha256 of the text the vector was computed from: a "Przedmiot /
    -- Sekcja" header plus content, so identical copies (PE) share one vector.
    -- No fixed dimension: vectors are only compared within one
    -- embedding_model, and without an index nothing needs it.
    content_hash TEXT,
    embedding vector,
    embedding_model TEXT,
    last_updated TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_course_card_section_course FOREIGN KEY (course_code, major_id, semester)
        REFERENCES courses (course_code, major_id, semester) ON DELETE CASCADE,
    CONSTRAINT uk_course_card_sections UNIQUE (course_code, major_id, semester, section, chunk_index)
);
