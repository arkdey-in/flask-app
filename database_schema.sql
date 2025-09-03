-- database_schema.sql
CREATE DATABASE IF NOT EXISTS data_1;
USE data_1;

-- Super Admin Table
CREATE TABLE IF NOT EXISTS superadmin (
    superadmin_id INT AUTO_INCREMENT PRIMARY KEY,
    superadmin_name VARCHAR(100) NOT NULL,
    superadmin_email VARCHAR(100) UNIQUE NOT NULL,
    superadmin_password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Admin Table
CREATE TABLE IF NOT EXISTS admin (
    admin_id INT AUTO_INCREMENT PRIMARY KEY,
    admin_name VARCHAR(100) NOT NULL,
    admin_email VARCHAR(100) UNIQUE NOT NULL,
    admin_password VARCHAR(255) NOT NULL,
    superadmin_id INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (superadmin_id) REFERENCES superadmin(superadmin_id)
);

-- Categories Table
CREATE TABLE IF NOT EXISTS categories (
    c_id INT AUTO_INCREMENT PRIMARY KEY,
    category_name VARCHAR(100) UNIQUE NOT NULL,
    c_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sub Categories Table
CREATE TABLE IF NOT EXISTS sub_categories (
    sc_id INT AUTO_INCREMENT PRIMARY KEY,
    category_id INT NOT NULL,
    sub_category_name VARCHAR(100) NOT NULL,
    sc_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (category_id) REFERENCES categories(c_id),
    UNIQUE KEY unique_category_subcategory (category_id, sub_category_name)
);

-- Tags Table
CREATE TABLE IF NOT EXISTS tags (
    t_id INT AUTO_INCREMENT PRIMARY KEY,
    tag_name VARCHAR(100) UNIQUE NOT NULL,
    t_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Documents Table
CREATE TABLE IF NOT EXISTS documents (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    sub_category VARCHAR(100),
    tags TEXT,
    description TEXT,
    file_path TEXT NOT NULL,
    extracted_data JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Subadmin Roles Table
CREATE TABLE IF NOT EXISTS subadminroles (
    r_id INT AUTO_INCREMENT PRIMARY KEY,
    role_name VARCHAR(100) UNIQUE NOT NULL,
    permissions JSON,
    r_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Subadmin Table
CREATE TABLE IF NOT EXISTS subadmin (
    subadmin_id INT AUTO_INCREMENT PRIMARY KEY,
    subadmin_name VARCHAR(100) NOT NULL,
    subadmin_email VARCHAR(100) UNIQUE NOT NULL,
    subadmin_username VARCHAR(50) UNIQUE NOT NULL,
    subadmin_password VARCHAR(255) NOT NULL,
    role_id INT,
    u_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (role_id) REFERENCES subadminroles(r_id)
);

-- Deprecated: Admin Activities Table
-- Note: This table is no longer used by the new `log_user_activity` function.
-- I recommend migrating all data from this table and `super_admin_activities`
-- to the new `user_activities` table.
CREATE TABLE IF NOT EXISTS admin_activities (
    id INT AUTO_INCREMENT PRIMARY KEY,
    admin_id INT,
    admin_name VARCHAR(100),
    admin_ip VARCHAR(45),
    event_type VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,
    value TEXT,
    event_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (admin_id) REFERENCES admin(admin_id) ON DELETE SET NULL
);

-- Deprecated: Super Admin Activities Table
-- Note: This table is no longer used by the new `log_user_activity` function.
CREATE TABLE IF NOT EXISTS super_admin_activities (
    id INT AUTO_INCREMENT PRIMARY KEY,
    superadmin_id INT NOT NULL,
    superadmin_name VARCHAR(100) NOT NULL,
    superadmin_ip VARCHAR(45),
    event_type VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,
    value TEXT,
    event_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (superadmin_id) REFERENCES superadmin(superadmin_id) ON DELETE CASCADE
);

-- New, unified User Activities Table
CREATE TABLE IF NOT EXISTS user_activities (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    user_name VARCHAR(100) NOT NULL,
    user_type ENUM('Super Admin', 'Admin', 'Subadmin') NOT NULL,
    user_ip VARCHAR(45),
    event_type VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,
    value TEXT,
    event_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert initial super admin (use a real hashed password in production)
-- Replace the placeholder hash with a real one for security.
-- Use `python -c "import bcrypt; print(bcrypt.hashpw(b'your_password_here', bcrypt.gensalt()).decode('utf-8'))"`
INSERT IGNORE INTO superadmin (superadmin_name, superadmin_email, superadmin_password) 
VALUES ('Default Super Admin', 'superadmin@example.com', '$2b$12$R.vL.x8T7e1M4yF.X.n.w.u5c.rK.y.U.1c.h');

-- Insert default admin role with full permissions
INSERT IGNORE INTO subadminroles (role_name, permissions) VALUES (
    'Administrator', 
    '{
        "CATEGORIES": {"view_categories": "Yes", "create_category": "Yes", "edit_category": "Yes", "delete_category": "Yes"},
        "SUB_CATEGORIES": {"view_sub_categories": "Yes", "create_sub_category": "Yes", "edit_sub_category": "Yes", "delete_sub_category": "Yes"},
        "TAGS": {"view_tags": "Yes", "create_tag": "Yes", "edit_tag": "Yes", "delete_tag": "Yes"},
        "DOCUMENTS": {"view_documents": "Yes", "create_document": "Yes", "edit_document": "Yes", "delete_document": "Yes"},
        "ROLES": {"view_roles": "Yes", "create_role": "Yes", "edit_role": "Yes", "delete_role": "Yes"},
        "USERS": {"view_users": "Yes", "create_users": "Yes", "edit_users": "Yes", "delete_users": "Yes"},
        "USER_ACTIVITIES": {"view_user_activities": "Yes"},
        "DASHBOARDS": {"view_dashboard": "Yes"}
    }'
);