-- public.jobs definition

CREATE TABLE public.jobs (
    id bigserial NOT NULL,
    slug varchar(20) NULL,
    title_en varchar(250) DEFAULT ''::character varying NOT NULL,
    title_cn varchar(500) NOT NULL,
    description_en text NOT NULL,
    description_cn text NOT NULL,
    apply_urls json NULL,
    published_at timestamptz(3) NULL,
    job_type_id int4 NOT NULL,
    remote bool NULL,
    company_id int8 NULL,
    city_id int4 NULL,
    created_at timestamptz(3) NOT NULL,
    updated_at timestamptz(3) NOT NULL,
    salary_lower int8 NOT NULL,
    salary_upper int8 NOT NULL,
    salary_currency varchar(3) NOT NULL,
    salary_payroll_cycle varchar(20) NOT NULL,
    working_hours int4 NULL,
    working_time varchar(100) DEFAULT ''::character varying NULL,
    flex_working_time bool NULL,
    work_overtime bool NULL,
    annual_leave_days int4 NULL,
    bonus bool NULL,
    probation_months int4 NULL,
    salary_lower_cny int8 NULL,
    salary_upper_cny int8 NULL,
    similar_job_ids varchar(300) NULL,
    origin int2 DEFAULT '0'::smallint NULL,
    recommended bool DEFAULT false NOT NULL,
    created_by int8 DEFAULT '1'::bigint NOT NULL,
    status varchar(10) DEFAULT 'SAVED'::bpchar NOT NULL,
    safe_guaranteed bool DEFAULT false NULL,
    description_format varchar(8) DEFAULT 'TEXT'::character varying NOT NULL,
    pinned bool DEFAULT false NOT NULL,
    reason varchar(300) DEFAULT ''::character varying NOT NULL,
    remark varchar(512) DEFAULT ''::character varying NOT NULL,
    preferred_cv_lang varchar(20) DEFAULT 'any'::character varying NOT NULL,
    CONSTRAINT "jobsPRIMARY16" PRIMARY KEY (id)
);

CREATE INDEX city_id36 ON public.jobs USING btree (city_id);
CREATE INDEX company_id31 ON public.jobs USING btree (company_id);
CREATE INDEX created_at_index29 ON public.jobs USING btree (created_at);
CREATE INDEX created_by32 ON public.jobs USING btree (created_by);
CREATE INDEX fdx_title_cn33 ON public.jobs USING btree (title_cn);
CREATE INDEX "idx_publishedAt_status_pinned30" ON public.jobs USING btree (published_at, status, pinned);
CREATE INDEX idx_status28 ON public.jobs USING btree (status);
CREATE INDEX jobs_ibfk_334 ON public.jobs USING btree (job_type_id);
CREATE UNIQUE INDEX slug35 ON public.jobs USING btree (slug);

-- Foreign keys
ALTER TABLE public.jobs ADD CONSTRAINT fk_jobs_city_id FOREIGN KEY (city_id) REFERENCES public.cities(id);
ALTER TABLE public.jobs ADD CONSTRAINT fk_jobs_company_id FOREIGN KEY (company_id) REFERENCES public.companies(id);
ALTER TABLE public.jobs ADD CONSTRAINT fk_jobs_job_type_id FOREIGN KEY (job_type_id) REFERENCES public.job_types(id);

-- Field descriptions:
-- id: Unique job identifier
-- slug: URL-friendly job identifier
-- title_cn/title_en: Job title in Chinese/English
-- description_cn/description_en: Job description in Chinese/English
-- published_at: Publication timestamp (use for sorting recent jobs)
-- remote: Whether the job supports remote work
-- company_id: Foreign key to companies table
-- city_id: Foreign key to cities table
-- job_type_id: Foreign key to job_types table
-- salary_lower/salary_upper: Salary range
-- salary_currency: Currency code (e.g., 'CNY', 'USD')
-- salary_payroll_cycle: Payment cycle (e.g., 'monthly', 'yearly')
-- status: Job status ('PUBLISHED', 'SAVED', 'ARCHIVED', etc.)
-- recommended: Whether the job is recommended
-- pinned: Whether the job is pinned to top
