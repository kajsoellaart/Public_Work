class DatasetSchema:
    # === Identifiers (Nominal) ===
    id_columns = [
        "srch_id",
        "site_id",
        "visitor_location_country_id",
        "prop_country_id",
        "prop_id",
        "srch_destination_id"
    ]

    # === Date/Time (Ordinal / Interval depending on use) ===
    datetime_columns = [
        "date_time",
        "search_year",    # Ordinal – years are ordered but interval consistency may not hold (e.g., trends change per year)
        "search_month",   # Ordinal – months are ordered (1–12), but not interval (Dec to Jan ≠ Jan to Feb)
        "search_day"      # Ordinal – days are ordered (1–31), but spacing is not always uniform
    ]

    # === Categorical: Binary (Nominal, Boolean-encoded) ===
    binary_nominal_columns = [
        "prop_brand_bool",
        "promotion_flag",
        "srch_saturday_night_bool",
        "random_bool",
        "click_bool",
        "booking_bool"
    ]

    # === Categorical: Ordinal (Ranking or ordered categories) ===
    ordinal_columns = [
        "prop_starrating",
        "position",
        "srch_length_of_stay",
        "srch_booking_window",
        "srch_adults_count",
        "srch_children_count",
        "srch_room_count"
    ]

    # === Numerical: Interval/Ratio (Real-valued or Count) ===
    numerical_columns = [
        "visitor_hist_starrating",
        "visitor_hist_adr_usd",
        "prop_review_score",
        "prop_location_score1",
        "prop_location_score2",
        "prop_log_historical_price",
        "price_usd",
        "srch_query_affinity_score",
        "orig_destination_distance",
        "gross_bookings_usd",
        # Competitor-related numerical columns
        "comp1_rate", "comp1_inv", "comp1_rate_percent_diff",
        "comp2_rate", "comp2_inv", "comp2_rate_percent_diff",
        "comp3_rate", "comp3_inv", "comp3_rate_percent_diff",
        "comp4_rate", "comp4_inv", "comp4_rate_percent_diff",
        "comp5_rate", "comp5_inv", "comp5_rate_percent_diff",
        "comp6_rate", "comp6_inv", "comp6_rate_percent_diff",
        "comp7_rate", "comp7_inv", "comp7_rate_percent_diff",
        "comp8_rate", "comp8_inv", "comp8_rate_percent_diff"
    ]


    columns_to_drop = [
        "comp1_rate", "comp1_inv", "comp1_rate_percent_diff",
        "comp2_rate", "comp2_inv", "comp2_rate_percent_diff",
        "comp3_rate", "comp3_inv", "comp3_rate_percent_diff",
        "comp4_rate", "comp4_inv", "comp4_rate_percent_diff",
        "comp5_rate", "comp5_inv", "comp5_rate_percent_diff",
        "comp6_rate", "comp6_inv", "comp6_rate_percent_diff",
        "comp7_rate", "comp7_inv", "comp7_rate_percent_diff",
        "comp8_rate", "comp8_inv", "comp8_rate_percent_diff"
    ]

    columns_zero_should_be_nan = [

    ]