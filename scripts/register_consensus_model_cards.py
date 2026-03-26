import logging
import sys
from datetime import datetime, timezone

from src.db.phase2_recorder import Phase2Recorder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Connecting to Phase 2 Database...")
    recorder = Phase2Recorder(bootstrap_schema=True)

    now = datetime.now(timezone.utc)

    # 1. Register Weighted Consensus Model
    weighted_card = {
        "model_id": "consensus_weighted_v1",
        "agent": "consensus",
        "model_family": "rule_based_weighted",
        "version": "1.0",
        "created_at": now,
        "updated_at": now,
        "status": "active",
        "design_notes": "Static weight combination (Technical=0.5, Regime=0.3, Sentiment=0.2) with Crisis routing.",
        "features": ["technical_prediction", "regime_prediction", "sentiment_score"],
        "labels": ["direction"],
    }
    
    logger.info("Registering model card: consensus_weighted_v1")
    recorder.save_model_card(weighted_card)

    # 2. Register Challenger Consensus Model
    challenger_card = {
        "model_id": "consensus_challenger_v1",
        "agent": "consensus",
        "model_family": "bayesian_ridge_estar",
        "version": "1.0",
        "created_at": now,
        "updated_at": now,
        "status": "active",
        "design_notes": "Learned ensemble weights using Bayesian Ridge Regression with LSTAR/ESTAR dynamic shifts for regime volatility.",
        "features": ["technical_prediction", "regime_prediction", "sentiment_score"],
        "labels": ["direction"],
    }
    
    logger.info("Registering model card: consensus_challenger_v1")
    recorder.save_model_card(challenger_card)

    logger.info("Successfully registered consensus model cards in the DB.")

if __name__ == "__main__":
    main()
