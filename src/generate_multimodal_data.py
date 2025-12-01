import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


# ==========================
# Config
# ==========================
RNG = np.random.default_rng(42)
N_USERS = 20000
N_APPS = 5
DATA_DIR = Path("data")


def sample_from_probs(options, probs, size):
    """Helper to sample categorical arrays with given probabilities."""
    return RNG.choice(options, p=probs, size=size)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Simulating {N_USERS} users across {N_APPS} apps...\n")

    countries = ["AT", "DE", "ES", "IT", "FR"]
    country_probs = np.array([0.15, 0.25, 0.25, 0.2, 0.15])

    devices = ["ios", "android"]
    device_probs = np.array([0.45, 0.55])

    install_sources = ["ads", "organic", "cross_promo"]
    install_source_probs = np.array([0.5, 0.3, 0.2])

    segments = ["low", "mid", "high"]
    segment_probs = np.array([0.6, 0.3, 0.1])

    base_date = datetime(2025, 1, 1)

    user_rows = []
    event_rows = []

    for user_id in range(1, N_USERS + 1):
        # -------- Static features --------
        country = sample_from_probs(countries, country_probs, 1)[0]
        device = sample_from_probs(devices, device_probs, 1)[0]
        install_source = sample_from_probs(install_sources, install_source_probs, 1)[0]
        segment = sample_from_probs(segments, segment_probs, 1)[0]

        age = int(RNG.normal(30, 7))
        age = int(np.clip(age, 18, 65))

        days_since_install = int(RNG.integers(1, 60))  # hasta 2 meses
        install_date = base_date - timedelta(days=days_since_install)

        # segment → engagement & spend propensities
        if segment == "low":
            lambda_sessions = RNG.uniform(2, 6)
            spend_prob = RNG.uniform(0.05, 0.15)
            spend_scale = RNG.uniform(1.0, 3.0)
        elif segment == "mid":
            lambda_sessions = RNG.uniform(5, 15)
            spend_prob = RNG.uniform(0.15, 0.35)
            spend_scale = RNG.uniform(2.0, 6.0)
        else:  # high
            lambda_sessions = RNG.uniform(10, 30)
            spend_prob = RNG.uniform(0.3, 0.6)
            spend_scale = RNG.uniform(5.0, 15.0)

        # country/device tweaks
        if country in ["DE", "FR"]:
            lambda_sessions *= 1.05
        if device == "ios":
            spend_scale *= 1.1

        n_sessions_30d = int(RNG.poisson(lambda_sessions))
        n_sessions_30d = max(n_sessions_30d, 0)

        session_lengths = []
        session_revenues = []
        session_days = []

        # -------- Session-level "events" (sequence) --------
        for s_idx in range(n_sessions_30d):
            # day offset inside 30-day window
            day_offset = int(RNG.integers(0, 30))
            event_time = install_date + timedelta(days=day_offset)

            app_id = int(RNG.integers(1, N_APPS + 1))

            # session length in seconds (gamma-like)
            sess_len = float(RNG.gamma(shape=2.0, scale=120.0))  # ~ mean 240s
            sess_len = max(sess_len, 10.0)

            # revenue event
            if RNG.random() < spend_prob:
                revenue = float(RNG.exponential(spend_scale))
            else:
                revenue = 0.0

            session_lengths.append(sess_len)
            session_revenues.append(revenue)
            session_days.append(day_offset)

            event_rows.append(
                {
                    "user_id": user_id,
                    "session_index": s_idx,
                    "timestamp": event_time.isoformat(),
                    "day_since_install": day_offset,
                    "app_id": app_id,
                    "session_length": sess_len,
                    "revenue": revenue,
                    "country": country,
                    "device": device,
                }
            )

        total_sessions = len(session_lengths)
        total_revenue_30d = float(np.sum(session_revenues)) if total_sessions > 0 else 0.0
        avg_session_length = float(np.mean(session_lengths)) if total_sessions > 0 else 0.0

        # "True" LTV: revenue + expected future value according to segment
        if segment == "low":
            future_factor = RNG.uniform(1.1, 1.3)
        elif segment == "mid":
            future_factor = RNG.uniform(1.2, 1.6)
        else:
            future_factor = RNG.uniform(1.4, 2.0)

        ltv_true_30d = total_revenue_30d * future_factor

        # Return after 7 days: if there is a session after day 7
        has_late_session = any(np.array(session_days) >= 7) if total_sessions > 0 else False
        return_7d = int(has_late_session)

        # Censoring: some users we only observed partially
        is_censored = int(RNG.random() < 0.2)  # 20% censored
        if is_censored:
            observed_factor = RNG.uniform(0.4, 0.9)
            ltv_observed_30d = ltv_true_30d * observed_factor
        else:
            ltv_observed_30d = ltv_true_30d

        user_rows.append(
            {
                "user_id": user_id,
                "country": country,
                "device": device,
                "install_source": install_source,
                "segment": segment,
                "age": age,
                "days_since_install": days_since_install,
                "total_sessions_30d": total_sessions,
                "total_revenue_30d": total_revenue_30d,
                "avg_session_length": avg_session_length,
                "ltv_true_30d": ltv_true_30d,
                "ltv_observed_30d": ltv_observed_30d,
                "return_7d": return_7d,
                "is_censored": is_censored,
            }
        )

        if user_id % 5000 == 0:
            print(f"  ... simulated {user_id} users")

    users_df = pd.DataFrame(user_rows)
    events_df = pd.DataFrame(event_rows)

    # Save CSV + Parquet
    users_csv = DATA_DIR / "users_ltv.csv"
    events_csv = DATA_DIR / "events_sessions.csv"
    users_parquet = DATA_DIR / "users_ltv.parquet"
    events_parquet = DATA_DIR / "events_sessions.parquet"

    users_df.to_csv(users_csv, index=False)
    events_df.to_csv(events_csv, index=False)
    users_df.to_parquet(users_parquet, index=False)
    events_df.to_parquet(events_parquet, index=False)

    print("\n✅ Dataset generated:")
    print(f"   Users:   {len(users_df)}  → {users_csv.name}, {users_parquet.name}")
    print(f"   Events:  {len(events_df)}  → {events_csv.name}, {events_parquet.name}")
    print("\nExample user row:")
    print(users_df.head(3))
    print("\nExample event row:")
    print(events_df.head(3))


if __name__ == "__main__":
    main()

