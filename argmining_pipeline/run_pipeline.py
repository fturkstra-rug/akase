from .src.pipeline import run_pipeline
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("artifacts/logs/project.log"),
        logging.StreamHandler()
    ]
)

def main():
    logging.getLogger("run").info("Starting pipeline")
    run_pipeline(input_dir="owi_data")

if __name__ == "__main__":
    main()
