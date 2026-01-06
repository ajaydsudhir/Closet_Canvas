import multiprocessing
from services.ingest.worker import run_worker_service


def main() -> None:
    """Run gating, SMPL, and recommendation workers in parallel processes."""
    # Start gating worker in a separate process
    gating_process = multiprocessing.Process(
        target=run_worker_service,
        kwargs={"queue_name": "gating", "enable_listener": False},
        name="GatingWorker",
    )
    gating_process.start()
    print("Started gating worker process")

    # Start SMPL worker in a separate process
    smpl_process = multiprocessing.Process(
        target=run_worker_service,
        kwargs={"queue_name": "smpl", "enable_listener": False},
        name="SMPLWorker",
    )
    smpl_process.start()
    print("Started SMPL worker process")

    # Start recommendation worker in a separate process
    recommendation_process = multiprocessing.Process(
        target=run_worker_service,
        kwargs={"queue_name": "recommendation", "enable_listener": False},
        name="RecommendationWorker",
    )
    recommendation_process.start()
    print("Started recommendation worker process")

    # Wait for all processes
    try:
        gating_process.join()
        smpl_process.join()
        recommendation_process.join()
    except KeyboardInterrupt:
        print("\nShutting down workers...")
        gating_process.terminate()
        smpl_process.terminate()
        recommendation_process.terminate()
        gating_process.join()
        smpl_process.join()
        recommendation_process.join()


if __name__ == "__main__":
    main()
