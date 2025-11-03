import os
import re
import json
from typing import Optional
import asyncpg
from fastmcp import FastMCP, Context
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()


class DatabaseManager:
    """Manages database connections and query execution."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        """Establish database connection pool."""
        if not self.pool:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=1,
                max_size=5,
                command_timeout=60
            )

    async def execute_query(self, sql: str) -> list[dict]:
        """Execute a SELECT query and return results as list of dictionaries."""
        if not self.pool:
            await self.connect()

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql)
            return [dict(row) for row in rows]

    async def close(self) -> None:
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None


class SQLValidator:
    """Validates SQL queries for security and constraints."""

    def __init__(self, max_results: int = 100):
        self.max_results = max_results
        self.dangerous_keywords = [
            'DROP', 'DELETE', 'TRUNCATE', 'UPDATE', 'INSERT',
            'ALTER', 'CREATE', 'GRANT', 'REVOKE', 'EXECUTE',
            'CALL', 'DECLARE', 'EXEC'
        ]

    def validate_sql(self, sql: str) -> tuple[bool, str]:
        """
        Validate SQL query for security.
        Returns: (is_valid, error_message)
        """
        sql_upper = sql.upper().strip()

        # Check if it's a SELECT query
        if not self.is_select_only(sql_upper):
            return False, "Only SELECT queries are allowed"

        # Check for dangerous keywords
        is_safe, keyword = self.check_dangerous_keywords(sql_upper)
        if not is_safe:
            return False, f"Dangerous keyword detected: {keyword}"

        # Check for multiple statements (SQL injection attempt)
        if ';' in sql.rstrip(';'):
            return False, "Multiple SQL statements are not allowed"

        return True, ""

    def is_select_only(self, sql: str) -> bool:
        """Check if SQL is a SELECT query."""
        return sql.strip().startswith('SELECT')

    def check_dangerous_keywords(self, sql: str) -> tuple[bool, str]:
        """
        Check for dangerous SQL keywords.
        Returns: (is_safe, dangerous_keyword)
        """
        for keyword in self.dangerous_keywords:
            # Use word boundary to avoid false positives
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, sql):
                return False, keyword
        return True, ""

    def add_limit_if_needed(self, sql: str) -> str:
        """Add LIMIT clause if not present or if it exceeds max_results."""
        sql_upper = sql.upper()

        # If already has LIMIT
        if 'LIMIT' in sql_upper:
            # Extract the limit value
            match = re.search(r'LIMIT\s+(\d+)', sql_upper)
            if match:
                limit_value = int(match.group(1))
                if limit_value > self.max_results:
                    # Replace with max_results
                    sql = re.sub(
                        r'LIMIT\s+\d+',
                        f'LIMIT {self.max_results}',
                        sql,
                        flags=re.IGNORECASE
                    )
            return sql

        # Add LIMIT if not present
        sql = sql.rstrip(';')
        return f"{sql} LIMIT {self.max_results};"


class SchemaProvider:
    """Provides database schema information and query examples."""

    @staticmethod
    def get_table_schema() -> str:
        """Return the jobs table schema definition."""
        return """
## Jobs Table Schema

### Main Fields:
- id (bigint): Unique job identifier
- slug (varchar): URL-friendly job identifier
- title_cn (varchar): Job title in Chinese
- title_en (varchar): Job title in English
- description_cn (text): Job description in Chinese
- description_en (text): Job description in English
- published_at (timestamp): Publication timestamp
- status (varchar): Job status - use 'PUBLISHED' to get active jobs
- remote (boolean): Whether the job supports remote work

### Company & Location:
- company_id (bigint): Foreign key to companies table
- city_id (int): Foreign key to cities table
- job_type_id (int): Foreign key to job_types table

### Salary Information:
⚠️ 注意：根据实际数据库调整字段名
如果数据库使用 salary_lower/salary_upper：
- salary_lower (bigint): Lower bound of salary range
- salary_upper (bigint): Upper bound of salary range

如果数据库使用单个 salary 字段：
- salary (bigint): Salary amount
- salary_currency (varchar): Currency code (CNY, USD, etc.)
- salary_payroll_cycle (varchar): Payment cycle (monthly, yearly, etc.)

### Additional Features:
- working_hours (int): Work hours per week
- flex_working_time (boolean): Flexible working time
- work_overtime (boolean): Overtime required
- annual_leave_days (int): Annual leave days
- bonus (boolean): Has bonus
- recommended (boolean): Recommended job
- pinned (boolean): Pinned to top
"""

    @staticmethod
    def get_related_tables() -> str:
        """Return information about related tables."""
        return """
## Related Tables (for JOIN operations):

### companies table:
- id: Company identifier
- name: Company name
- (use: LEFT JOIN companies c ON j.company_id = c.id)

### cities table:
- id: City identifier
- name: City name
- (use: LEFT JOIN cities ct ON j.city_id = ct.id)

### job_types table:
- id: Job type identifier
- name: Job type name
- (use: LEFT JOIN job_types jt ON j.job_type_id = jt.id)
"""

    @staticmethod
    def get_query_examples() -> str:
        """Return typical query examples for common business scenarios."""
        return """
## Typical Query Examples:

### Scenario 1: Hot Jobs for Job Seekers
Query: "Show me the most recent high-salary jobs"

⚠️ 注意：根据实际数据库字段调整以下 SQL

如果使用 salary_lower/salary_upper：
```sql
SELECT
    j.title_cn,
    c.name as company_name,
    j.description_cn,
    j.salary_lower,
    j.salary_upper,
    j.salary_currency,
    j.published_at
FROM jobs j
LEFT JOIN companies c ON j.company_id = c.id
WHERE j.status = 'PUBLISHED'
ORDER BY j.published_at DESC, j.salary_upper DESC
LIMIT 20;
```

如果使用单个 salary 字段：
```sql
SELECT
    j.title_cn,
    c.name as company_name,
    j.description_cn,
    j.salary,
    j.salary_currency,
    j.published_at
FROM jobs j
LEFT JOIN companies c ON j.company_id = c.id
WHERE j.status = 'PUBLISHED'
ORDER BY j.published_at DESC, j.salary DESC
LIMIT 20;
```

### Scenario 2: Remote Jobs for Recruiters
Query: "Which jobs support remote work?"

如果使用 salary_lower/salary_upper：
```sql
SELECT
    j.id,
    j.title_cn,
    c.name as company_name,
    j.remote,
    j.salary_lower,
    j.salary_upper,
    j.salary_currency,
    j.description_cn
FROM jobs j
LEFT JOIN companies c ON j.company_id = c.id
WHERE j.remote = true AND j.status = 'PUBLISHED';
```

如果使用单个 salary 字段：
```sql
SELECT
    j.id,
    j.title_cn,
    c.name as company_name,
    j.remote,
    j.salary,
    j.salary_currency,
    j.description_cn
FROM jobs j
LEFT JOIN companies c ON j.company_id = c.id
WHERE j.remote = true AND j.status = 'PUBLISHED';
```

### Important Notes:
1. Always filter by status = 'PUBLISHED' to get active jobs
2. Use LEFT JOIN for company information
3. salary_lower and salary_upper define the salary range
4. Use ORDER BY published_at DESC for recent jobs
5. Consider sorting by salary for high-paying jobs
"""


class MCPTools:
    """MCP tools for SQL generation and execution."""

    def __init__(self, db_manager: DatabaseManager, validator: SQLValidator):
        self.db_manager = db_manager
        self.validator = validator
        self.schema_provider = SchemaProvider()

        # Initialize OpenAI client if API key is provided (for standalone mode)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL")

        self.openai_client = None
        if self.openai_api_key:
            self.openai_client = AsyncOpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url if self.openai_base_url else None
            )

    async def generate_sql(self, query: str, ctx: Context) -> str:
        """
        Generate SQL query from natural language using LLM.

        Supports two modes:
        1. MCP Mode: Uses ctx.sample() to request LLM from MCP client (e.g., Claude Desktop)
        2. Standalone Mode: Uses OpenAI API directly (requires OPENAI_API_KEY in .env)

        Args:
            query: Natural language query description
            ctx: FastMCP context for LLM sampling

        Returns:
            Generated SQL query
        """
        await ctx.info(f"Generating SQL for: {query}")

        # Construct the prompt for LLM
        system_prompt = f"""You are a PostgreSQL SQL expert. Generate a SQL query based on the user's natural language request.

{self.schema_provider.get_table_schema()}

{self.schema_provider.get_related_tables()}

{self.schema_provider.get_query_examples()}

IMPORTANT RULES:
1. ONLY generate SELECT queries
2. Always use table alias 'j' for jobs table
3. Use LEFT JOIN for related tables (companies, cities, job_types)
4. Filter by status = 'PUBLISHED' for active jobs
5. Consider using ORDER BY and LIMIT appropriately
6. Return ONLY the SQL query without explanations
7. Use appropriate Chinese fields (title_cn, description_cn) for Chinese queries
"""

        user_prompt = f"Generate a SQL query for this request: {query}"

        # Choose LLM mode: OpenAI API (standalone) or ctx.sample (MCP client)
        if self.openai_client:
            # Standalone mode: Use OpenAI API directly
            await ctx.info(f"Using OpenAI API (model: {self.openai_model})")
            try:
                response = await self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3
                )
                sql_text = response.choices[0].message.content
            except Exception as e:
                await ctx.info(f"OpenAI API error: {e}")
                raise ValueError(f"Failed to generate SQL using OpenAI API: {e}")
        else:
            # MCP mode: Request LLM from MCP client
            await ctx.info("Using MCP client's LLM (ctx.sample)")
            try:
                response = await ctx.sample(
                    messages=user_prompt,
                    system_prompt=system_prompt,
                    temperature=0.3,
                    max_tokens=500
                )
                sql_text = response.text
            except Exception as e:
                await ctx.info(f"MCP sampling error: {e}")
                raise ValueError(
                    "Failed to generate SQL: MCP client doesn't provide LLM. "
                    "Either use Claude Desktop or set OPENAI_API_KEY in .env for standalone mode."
                )

        # Extract SQL from response (remove markdown code blocks if present)
        sql = sql_text.strip()
        sql = re.sub(r'^```sql\n', '', sql)
        sql = re.sub(r'^```\n', '', sql)
        sql = re.sub(r'\n```$', '', sql)
        sql = sql.strip()

        await ctx.info(f"Generated SQL: {sql}")

        return sql

    async def execute_sql(self, sql: str, ctx: Context) -> str:
        """
        Execute SQL query with security validation.

        Args:
            sql: SQL query to execute
            ctx: FastMCP context for logging

        Returns:
            JSON string of query results
        """
        await ctx.info(f"Validating SQL query...")

        # Validate SQL
        is_valid, error_message = self.validator.validate_sql(sql)
        if not is_valid:
            await ctx.info(f"SQL validation failed: {error_message}")
            return json.dumps({
                "error": error_message,
                "status": "validation_failed"
            }, ensure_ascii=False, indent=2)

        # Add LIMIT if needed
        sql = self.validator.add_limit_if_needed(sql)
        await ctx.debug(f"Final SQL: {sql}")

        try:
            # Execute query
            await ctx.info("Executing query...")
            results = await self.db_manager.execute_query(sql)

            await ctx.info(f"Query successful. Returned {len(results)} rows.")

            return json.dumps({
                "status": "success",
                "row_count": len(results),
                "data": results
            }, ensure_ascii=False, indent=2, default=str)

        except Exception as e:
            error_msg = str(e)
            await ctx.info(f"Query execution failed: {error_msg}")
            return json.dumps({
                "error": error_msg,
                "status": "execution_failed"
            }, ensure_ascii=False, indent=2)


# Load configuration
database_url = os.getenv("DATABASE_URL", "")
max_results = int(os.getenv("MAX_QUERY_RESULTS", "100"))

# Initialize components
db_manager = DatabaseManager(database_url) if database_url else None
validator = SQLValidator(max_results=max_results)
mcp_tools = MCPTools(db_manager, validator) if db_manager else None

# Create FastMCP server
mcp = FastMCP("Query2SQL Service")


@mcp.tool
async def generate_sql(query: str, ctx: Context) -> str:
    """
    Generate SQL query from natural language description.

    This tool uses an LLM to convert natural language queries into SQL statements
    for the jobs database. It understands common business scenarios like finding
    hot jobs, remote positions, salary ranges, etc.

    Args:
        query: Natural language description of what you want to query

    Returns:
        A SQL SELECT query that matches the request

    Examples:
        - "Show me the latest high-salary jobs"
        - "Which jobs support remote work?"
        - "Find jobs in Beijing with salary above 20000"
    """
    if not mcp_tools:
        raise ValueError("Database not configured. Set DATABASE_URL environment variable.")
    return await mcp_tools.generate_sql(query, ctx)


@mcp.tool
async def execute_sql(sql: str, ctx: Context) -> str:
    """
    Execute a SQL query against the jobs database.

    This tool safely executes SELECT queries with built-in security validations.
    It prevents dangerous operations and limits result set size for safety.

    Args:
        sql: SQL SELECT query to execute

    Returns:
        JSON string containing query results or error message

    Security Features:
        - Only SELECT queries allowed
        - Blocks dangerous keywords (DROP, DELETE, etc.)
        - Limits maximum result rows
        - Prevents SQL injection

    Response Format:
        {
            "status": "success" | "validation_failed" | "execution_failed",
            "row_count": <number>,
            "data": [<row objects>],
            "error": "<error message if failed>"
        }
    """
    if not mcp_tools:
        raise ValueError("Database not configured. Set DATABASE_URL environment variable.")
    return await mcp_tools.execute_sql(sql, ctx)


if __name__ == "__main__":
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is required")

    # Check for command line arguments
    import sys

    # Default to stdio transport
    transport = "stdio"
    host = "127.0.0.1"
    port = 8000

    # Parse command line arguments for transport mode
    # Support --http, --streamable-http, or --transport=xxx
    if "--http" in sys.argv or "--transport=http" in sys.argv:
        transport = "http"
    elif "--streamable-http" in sys.argv or "--transport=streamable-http" in sys.argv:
        transport = "streamable-http"
    else:
        # Check for --transport with value
        for arg in sys.argv:
            if arg.startswith("--transport="):
                transport = arg.split("=", 1)[1]
                break

    # Check for host argument
    if "--host" in sys.argv:
        idx = sys.argv.index("--host")
        if idx + 1 < len(sys.argv):
            host = sys.argv[idx + 1]
    else:
        # Check for --host=xxx format
        for arg in sys.argv:
            if arg.startswith("--host="):
                host = arg.split("=", 1)[1]
                break

    # Check for port argument
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            port = int(sys.argv[idx + 1])
    else:
        # Check for --port=xxx format
        for arg in sys.argv:
            if arg.startswith("--port="):
                port = int(arg.split("=", 1)[1])
                break

    # Run the server
    if transport in ("http", "streamable-http", "sse"):
        print(f"Starting {transport.upper()} server on {host}:{port}")
        mcp.run(transport=transport, host=host, port=port)
    else:
        mcp.run()
