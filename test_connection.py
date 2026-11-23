import psycopg

from app.config import settings

if not settings.DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env file")

try:
    # Connect to Supabase PostgreSQL
    with psycopg.connect(settings.DATABASE_URL) as conn:

        with conn.cursor() as cursor:

            cursor.execute("""
                SELECT
                    version(),
                    current_database(),
                    current_user;
            """)

            result = cursor.fetchone()

            print("Successfully connected to Supabase!")
            print("Database:", result[1])
            print("User:", result[2])

except Exception as error:
    print("Database connection failed!")
    print("Error:", error)