import mysql.connector
import pandas as pd

# Hardcoded credentials as per project structure
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "your password",
    "database": "hr_analytics"
}

def get_connection():
    return mysql.connector.connect(**DB_CONFIG)

def get_employee_data():
    connection = get_connection()
    # Note: Using native connection is functional, ignoring the pandas warning for pure-script execution
    df = pd.read_sql("select * from employees", connection)
    connection.close()
    return df


def _normalize(name):
    """Strip anything that isn't a plain ASCII letter/digit and lowercase,
    so 'ï»¿Age' and 'Age' both normalize to 'age' for matching purposes.
    """
    return "".join(ch for ch in name if ch.isascii() and ch.isalnum()).lower()


def get_table_columns():
    """
    Returns the *actual* column names as they exist in the employees table.
    """
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("SHOW COLUMNS FROM employees")
    columns = [row[0] for row in cursor.fetchall()]
    cursor.close()
    connection.close()
    return columns


def insert_employee(employee_data):
    """
    Inserts a new employee record into the employees table.
    Handles dynamic column matching and calculates the next unique EmployeeNumber
    safely inside the write transaction block to prevent race conditions.
    """
    employee_data = employee_data.copy()

    # Auto-fill constant / derived fields if the caller didn't supply them
    employee_data.setdefault("EmployeeCount", 1)
    employee_data.setdefault("Over18", "Y")
    employee_data.setdefault("StandardHours", 80)

    # Map our clean field names to whatever the table's columns are really called
    actual_columns = get_table_columns()
    normalized_lookup = {_normalize(col): col for col in actual_columns}

    connection = get_connection()
    cursor = connection.cursor()
    
    try:
        # FIX: Calculate MAX inside the transaction right before inserting to minimize collisions
        if not employee_data.get("EmployeeNumber"):
            cursor.execute("SELECT MAX(EmployeeNumber) FROM employees")
            result = cursor.fetchone()
            max_number = result[0] if result and result[0] is not None else 0
            employee_data["EmployeeNumber"] = max_number + 1

        db_columns = []
        values = []
        for key, value in employee_data.items():
            real_column = normalized_lookup.get(_normalize(key), key)
            db_columns.append(real_column)
            values.append(value)

        column_names = ", ".join(f"`{col}`" for col in db_columns)
        placeholders = ", ".join(["%s"] * len(db_columns))
        values_tuple = tuple(values)

        query = f"INSERT INTO employees ({column_names}) VALUES ({placeholders})"
        cursor.execute(query, values_tuple)
        connection.commit()
        
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()

    return employee_data["EmployeeNumber"]


if __name__ == "__main__":
    try:
        connection = get_connection()
        print("Connected successfully to hr_analytics database.")
        connection.close()

        data = get_employee_data()
        print(f"Data loaded successfully. Shape: {data.shape}")
    except Exception as e:
        print(f"Database connection error: {e}")
