LOG="logs/preprocess_15771424.err"
OUTPUT_DIR="/gpfs/scratch1/shared/fturkstra/.owi/public/main_clean"

# Extract the failed input paths from the log
grep "Error processing" "$LOG" \
  | sed -E 's/.*Error processing (\/gpfs[^:]+):.*/\1/' > failed_input_files.txt

# Map them to output paths
sed "s|/main/|/main_clean/|" failed_input_files.txt > failed_output_files.txt

# Delete only those outputs
cat failed_output_files.txt | xargs -r rm -f
