import duckdb
from pathlib import Path

# Find all the downloaded parquet files.
files = [
    str(p) for p in Path("/gpfs/scratch1/shared/fturkstra/.owi/public/main").rglob("*.parquet")
    if p.stat().st_size > 1000  # skip tiny/empty files
]

print(f"Found {len(files)} valid parquet files.")

# Read all the parquet files into DuckDB.
con = duckdb.connect()
print("Connected to database, quack quack!")

# Check if the file is valid and whether it is allowed to be indexed.
total_object_count_query = f"SELECT COUNT(*) FROM read_parquet({files}) WHERE valid AND ows_index"
total = con.execute(total_object_count_query).fetchone()[0]
print(total)

# Some curlielabels start with /en/ and some do not.
# Documents can have multiple labels so we need to unnest them.
# Only check valid documents that are allowed to be indexed.
count_curlielabels_query = """
SELECT 
    CASE
        WHEN label LIKE '/en/%' THEN SUBSTR(label, 5)  -- strip leading '/en/'
        ELSE label
    END AS clean_label,
    COUNT(*) AS count
FROM (
    SELECT UNNEST(curlielabels_en) AS label
    FROM read_parquet($files)
    WHERE valid AND ows_index
)
GROUP BY clean_label
ORDER BY count DESC
"""

# Save the curlielabel counts to a csv file.
label_counts = con.execute(count_curlielabels_query, {"files": files}).fetchdf()
label_counts.to_csv('curlielabel_counts.csv', index=False)

# The curlielabels have multiple levels of granularity, e.g. Computers/Hardware/Storage.
# Below we only count the top category, in the example above 'Computers'.
label_counts["top_category"] = label_counts["clean_label"].str.split("/").str[0]
top_level_counts = (
    label_counts.groupby("top_category", as_index=False)["count"]
    .sum()
    .sort_values("count", ascending=False)
)
print(top_level_counts)
