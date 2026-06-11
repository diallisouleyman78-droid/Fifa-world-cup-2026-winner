import streamlit as st


FLAG_EMOJIS = {
    "Argentina": "🇦🇷",
    "Australia": "🇦🇺",
    "Austria": "🇦🇹",
    "Belgium": "🇧🇪",
    "Brazil": "🇧🇷",
    "Canada": "🇨🇦",
    "Colombia": "🇨🇴",
    "Croatia": "🇭🇷",
    "Curaçao": "🇨🇼",
    "Czech Republic": "🇨🇿",
    "DR Congo": "🇨🇩",
    "Ecuador": "🇪🇨",
    "Egypt": "🇪🇬",
    "England": "🏴",
    "France": "🇫🇷",
    "Germany": "🇩🇪",
    "Haiti": "🇭🇹",
    "Iran": "🇮🇷",
    "Iraq": "🇮🇶",
    "Italy": "🇮🇹",
    "Japan": "🇯🇵",
    "Mexico": "🇲🇽",
    "Morocco": "🇲🇦",
    "Netherlands": "🇳🇱",
    "Norway": "🇳🇴",
    "Panama": "🇵🇦",
    "Paraguay": "🇵🇾",
    "Portugal": "🇵🇹",
    "Qatar": "🇶🇦",
    "Saudi Arabia": "🇸🇦",
    "Scotland": "🏴",
    "Senegal": "🇸🇳",
    "South Africa": "🇿🇦",
    "South Korea": "🇰🇷",
    "Spain": "🇪🇸",
    "Switzerland": "🇨🇭",
    "Tunisia": "🇹🇳",
    "Turkey": "🇹🇷",
    "Ukraine": "🇺🇦",
    "United States": "🇺🇸",
    "Uruguay": "🇺🇾",
    "Uzbekistan": "🇺🇿",
}


def flag(team):
    """Return flag emoji for a team."""
    return FLAG_EMOJIS.get(team, "🏳️")


def team_label(team):
    """Return team name with flag."""
    return f"{flag(team)} {team}"


def apply_global_styles():
    """Apply simple World Cup style CSS."""

    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #07111f 0%, #0b1f18 45%, #07111f 100%);
            color: #ffffff;
        }

        .main-title {
            font-size: 42px;
            font-weight: 800;
            margin-bottom: 0px;
        }

        .sub-title {
            font-size: 18px;
            color: #d5d5d5;
            margin-bottom: 24px;
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 16px;
            padding: 18px;
            margin-bottom: 14px;
        }

        .winner-card {
            background: linear-gradient(135deg, #f7d774, #b8860b);
            color: #06120f;
            border-radius: 20px;
            padding: 28px;
            text-align: center;
            font-size: 28px;
            font-weight: 800;
            margin: 20px 0;
        }

        .team-card {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 12px 14px;
            margin-bottom: 8px;
            border-left: 5px solid #30d158;
        }

        .round-title {
            font-size: 24px;
            font-weight: 700;
            margin-top: 22px;
            margin-bottom: 12px;
        }

        .small-muted {
            color: #cccccc;
            font-size: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_probability(value):
    """Format a probability as a percent string."""
    return f"{value * 100:.1f}%"