import sqlite3
import pandas as pd

DB_PATH = "data/northwind.sqlite"

class SQLiteTool:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path

    def get_schema(self, table_names=None):
        """Returns schema for specific tables or all key tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Canonical tables as per assignment
        canonical_tables = ["Orders", "Order Details", "Products", "Customers", "Categories", "Suppliers"]
        target_tables = table_names if table_names else canonical_tables
        
        schema_str = ""
        for table in target_tables:
            # Handle quoted table names for "Order Details"
            q_table = f'"{table}"' if " " in table else table
            
            try:
                cursor.execute(f"PRAGMA table_info({q_table})")
                columns = cursor.fetchall()
                if columns:
                    col_str = ", ".join([f"{c[1]} ({c[2]})" for c in columns])
                    schema_str += f"Table {table}: {col_str}\n"
            except Exception:
                continue
                
        conn.close()
        return schema_str

    def execute_query(self, sql):
        """Executes SQL and returns (columns, rows, error)."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            conn.close()
            return columns, rows, None
        except Exception as e:
            conn.close()
            return [], [], str(e)
