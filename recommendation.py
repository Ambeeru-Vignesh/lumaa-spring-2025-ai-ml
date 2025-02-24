# Import required libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import re

class EnhancedMovieRecommender:
    """
    An advanced content-based movie recommendation system that considers
    multiple factors including mood, time period, and themes.
    """

    @staticmethod
    def load_local_dataset():
        """Load the IMDB movies dataset from a local CSV file."""
        try:
            print("Loading local dataset...")
            df = pd.read_csv("imdb_top_1000.csv")
            print("Successfully loaded local dataset")
            return df
        except Exception as e:
            print(f"Error loading local dataset: {e}")
            print("Loading fallback sample dataset...")
            return pd.DataFrame({
                'Series_Title': ['Sample Movie 1', 'Sample Movie 2'],
                'Overview': ['Sample overview 1', 'Sample overview 2'],
                'Genre': ['Action, Adventure', 'Drama, Comedy'],
                'Director': ['Director 1', 'Director 2'],
                'Star1': ['Actor 1', 'Actor 3'],
                'Star2': ['Actor 2', 'Actor 4'],
                'Released_Year': [2020, 2021],
                'IMDB_Rating': [7.5, 8.0]
            })

    def __init__(self):
        """Initialize the recommender system."""
        self.df = self.load_local_dataset()
        self.mood_weight = 0.35
        self.theme_weight = 0.35
        self.base_weight = 0.30
        self.initialize_features()
        self.prepare_data()

    def initialize_features(self):
        """Initialize the system's understanding of moods and themes."""
        self.mood_indicators = {
            'cheerful': ['funny', 'humorous', 'light-hearted', 'upbeat', 'comedic'],
            'intense': ['thrilling', 'suspenseful', 'action-packed', 'exciting'],
            'emotional': ['touching', 'moving', 'dramatic', 'romantic'],
            'dark': ['gritty', 'noir', 'mysterious', 'supernatural'],
            'thoughtful': ['philosophical', 'psychological', 'thought-provoking']
        }

        self.theme_categories = {
            'transformation': ['journey', 'growth', 'change', 'discovery'],
            'conflict': ['war', 'battle', 'fight', 'rivalry', 'competition'],
            'relationships': ['family', 'friendship', 'love', 'betrayal'],
            'survival': ['apocalypse', 'disaster', 'wilderness', 'isolation'],
            'justice': ['revenge', 'law', 'crime', 'truth', 'investigation']
        }

        self.time_periods = {
            'contemporary': ['modern', 'present-day', 'current'],
            'retro': ['80s', '90s', 'vintage', 'classic'],
            'futuristic': ['future', 'cyberpunk', 'post-apocalyptic'],
            'period': ['historical', 'ancient', 'medieval', 'renaissance']
        }

    def prepare_data(self):
        """Prepare and enhance the movie data for recommendation."""
        self.df['enhanced_text'] = self.df.apply(self._create_enhanced_text, axis=1)
        self.vectorizer = TfidfVectorizer(
            max_features=4000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['enhanced_text'])

    def _create_enhanced_text(self, row):
        """Create enriched text representation for each movie."""
        components = [
            str(row['Overview']).lower() * 2,
            str(row['Genre']).lower() * 3,
            str(row['Director']).lower(),
            ' '.join([str(row['Star1']), str(row['Star2'])]).lower()
        ]
        return ' '.join(filter(None, components))

    def analyze_user_preferences(self, user_input):
        """Extract user preferences from input text."""
        user_input = user_input.lower()
        preferences = {
            'moods': self._extract_preferences(user_input, self.mood_indicators),
            'themes': self._extract_preferences(user_input, self.theme_categories),
            'time_period': self._extract_preferences(user_input, self.time_periods)
        }
        return preferences

    def _extract_preferences(self, text, category_dict):
        """Extract matching preferences from text for a given category."""
        matches = []
        for category, indicators in category_dict.items():
            if any(indicator in text for indicator in indicators):
                matches.append(category)
        return matches

    def calculate_preference_score(self, movie_text, preferences):
        """Calculate preference matching score for a movie."""
        mood_score = sum(1 for mood in preferences['moods']
                         if any(indicator in movie_text
                                for indicator in self.mood_indicators.get(mood, [])))

        theme_score = sum(1 for theme in preferences['themes']
                          if any(indicator in movie_text
                                 for indicator in self.theme_categories.get(theme, [])))

        return (mood_score / (len(preferences['moods']) or 1) * self.mood_weight +
                theme_score / (len(preferences['themes']) or 1) * self.theme_weight)

    def get_recommendations(self, user_input, n_recommendations=5):
        """Generate movie recommendations based on user input."""
        cleaned_input = re.sub(r'[^a-zA-Z\s]', ' ', user_input.lower())
        user_vector = self.vectorizer.transform([cleaned_input])

        base_similarities = cosine_similarity(user_vector, self.tfidf_matrix).flatten()
        preferences = self.analyze_user_preferences(user_input)

        final_scores = np.zeros_like(base_similarities)
        for idx, movie_text in enumerate(self.df['enhanced_text']):
            preference_score = self.calculate_preference_score(movie_text, preferences)
            final_scores[idx] = (base_similarities[idx] * self.base_weight +
                                 preference_score)

        top_indices = final_scores.argsort()[-n_recommendations:][::-1]

        recommendations = []
        for idx in top_indices:
            movie = self.df.iloc[idx]
            recommendations.append({
                'title': movie['Series_Title'],
                'year': movie['Released_Year'],
                'genre': movie['Genre'],
                'rating': movie['IMDB_Rating'],
                'synopsis': movie['Overview'],
                'match_score': round(final_scores[idx], 3)
            })

        return recommendations

def main():
    """Main function to demonstrate the recommendation system."""
    recommender = EnhancedMovieRecommender()
    print("\nWelcome to the Enhanced Movie Recommender!")
    while True:
        print("\nDescribe what kind of movie you're in the mood for (type 'exit' to quit):")
        user_input = input("> ")
        if user_input.strip().lower() == 'exit':
            print("Thank you for using the Enhanced Movie Recommender. Goodbye!")
            break
        recommendations = recommender.get_recommendations(user_input)

        print("\nTop Recommendations Based on Your Preferences:")
        print("-" * 50)
        for i, movie in enumerate(recommendations, 1):
            print(f"\n{i}. {movie['title']} ({movie['year']})")
            print(f"   Genre: {movie['genre']}")
            print(f"   IMDB Rating: {movie['rating']}")
            print(f"   Match Score: {movie['match_score']}")
            print(f"   Synopsis: {movie['synopsis'][:150]}...")

if __name__ == "__main__":
    main()