from services.ingest.worker import run_worker_service


def main() -> None:
    run_worker_service(
        queue_name="video",  # explicit queue name for clarity
        enable_listener=True,
    )


if __name__ == "__main__":
    main()
