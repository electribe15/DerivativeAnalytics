CREATE TABLE regulations (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    source VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE company_policies (
    id SERIAL PRIMARY KEY,
    company_name VARCHAR(255) NOT NULL,
    policy_title VARCHAR(255) NOT NULL,
    policy_document TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE compliance_reports (
    id SERIAL PRIMARY KEY,
    company_id INT REFERENCES company_policies(id),
    regulation_id INT REFERENCES regulations(id),
    report_date DATE NOT NULL,
    compliance_status VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE esg_metrics (
    id SERIAL PRIMARY KEY,
    report_id INT REFERENCES compliance_reports(id),
    metric_name VARCHAR(255) NOT NULL,
    metric_value FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);