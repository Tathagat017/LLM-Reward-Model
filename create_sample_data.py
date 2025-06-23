import pandas as pd
import random

def create_sample_training_data():
    """Create sample training data with pre-defined ranks for demonstration"""
    
    # Sample data with different quality levels
    sample_data = [
        # Prompt 1: Programming joke
        {
            'prompt': 'Tell me a funny joke about programming',
            'answer': 'Why do programmers prefer dark mode? Because light attracts bugs!',
            'rank': 1
        },
        {
            'prompt': 'Tell me a funny joke about programming',
            'answer': 'A programmer is told to go to the store and buy a gallon of milk, and if there are eggs, buy a dozen. He comes back with 12 gallons of milk.',
            'rank': 2
        },
        {
            'prompt': 'Tell me a funny joke about programming',
            'answer': 'Programming is hard. Sometimes it works, sometimes it doesn\'t.',
            'rank': 3
        },
        {
            'prompt': 'Tell me a funny joke about programming',
            'answer': 'Code code code.',
            'rank': 4
        },
        
        # Prompt 2: Renewable energy summary
        {
            'prompt': 'Write a brief summary of the benefits of renewable energy',
            'answer': 'Renewable energy offers numerous benefits including environmental protection through reduced greenhouse gas emissions, economic advantages through job creation and energy independence, and long-term sustainability. Solar, wind, and hydroelectric power provide clean alternatives to fossil fuels while becoming increasingly cost-effective.',
            'rank': 1
        },
        {
            'prompt': 'Write a brief summary of the benefits of renewable energy',
            'answer': 'Renewable energy is good for the environment and can save money. It includes solar and wind power which don\'t pollute as much as coal and oil.',
            'rank': 2
        },
        {
            'prompt': 'Write a brief summary of the benefits of renewable energy',
            'answer': 'Solar panels and wind turbines make electricity from sun and wind.',
            'rank': 3
        },
        {
            'prompt': 'Write a brief summary of the benefits of renewable energy',
            'answer': 'Energy is renewable.',
            'rank': 4
        },
        
        # Prompt 3: Importance of reading
        {
            'prompt': 'Compose a short essay about the importance of reading books',
            'answer': 'Reading books is fundamental to personal and intellectual development. It expands vocabulary, improves critical thinking skills, and provides exposure to diverse perspectives and ideas. Through reading, we can explore different worlds, learn from historical experiences, and develop empathy by understanding various characters and situations. Regular reading also enhances concentration, reduces stress, and serves as a gateway to lifelong learning.',
            'rank': 1
        },
        {
            'prompt': 'Compose a short essay about the importance of reading books',
            'answer': 'Reading books is important because it helps you learn new things and improves your vocabulary. Books can teach you about history, science, and different cultures. Reading also helps you relax and is a good hobby.',
            'rank': 2
        },
        {
            'prompt': 'Compose a short essay about the importance of reading books',
            'answer': 'Books are good. You should read them because they have words and stories.',
            'rank': 3
        },
        {
            'prompt': 'Compose a short essay about the importance of reading books',
            'answer': 'Read books.',
            'rank': 4
        },
        
        # Prompt 4: Perfect coffee
        {
            'prompt': 'Explain how to make a perfect cup of coffee',
            'answer': 'To make perfect coffee: 1) Use freshly ground, high-quality beans (medium grind for drip coffee). 2) Heat water to 195-205°F. 3) Use a ratio of 1:15 to 1:17 (coffee to water). 4) Pour hot water over grounds in circular motions, allowing 4-6 minutes brewing time. 5) Serve immediately in a pre-warmed cup. The key is using fresh beans, proper water temperature, and correct timing.',
            'rank': 1
        },
        {
            'prompt': 'Explain how to make a perfect cup of coffee',
            'answer': 'Use good coffee beans and hot water. Grind the beans fresh and brew for about 4 minutes. The water should be hot but not boiling.',
            'rank': 2
        },
        {
            'prompt': 'Explain how to make a perfect cup of coffee',
            'answer': 'Put coffee in hot water and wait.',
            'rank': 3
        },
        {
            'prompt': 'Explain how to make a perfect cup of coffee',
            'answer': 'Coffee hot.',
            'rank': 4
        },
        
        # Prompt 5: Good friend qualities
        {
            'prompt': 'Describe what makes a good friend',
            'answer': 'A good friend demonstrates trustworthiness, loyalty, and genuine care for your well-being. They listen actively without judgment, offer support during difficult times, and celebrate your successes. Good friends are reliable, honest even when it\'s difficult, and respect your boundaries. They maintain confidentiality, show empathy, and invest time and effort in the relationship. Most importantly, they accept you for who you are while encouraging your growth.',
            'rank': 1
        },
        {
            'prompt': 'Describe what makes a good friend',
            'answer': 'A good friend is someone who is trustworthy, kind, and supportive. They listen to you and help when you need it. Good friends are also fun to be around and share common interests.',
            'rank': 2
        },
        {
            'prompt': 'Describe what makes a good friend',
            'answer': 'Friends are nice and helpful.',
            'rank': 3
        },
        {
            'prompt': 'Describe what makes a good friend',
            'answer': 'Friend good.',
            'rank': 4
        }
    ]
    
    return sample_data

def main():
    """Create and save sample training data"""
    print("Creating sample training data...")
    
    sample_data = create_sample_training_data()
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Save to CSV
    df.to_csv('training_data.csv', index=False)
    
    print(f"Sample training data created with {len(sample_data)} examples")
    print("Data saved to 'training_data.csv'")
    print("\nData distribution by rank:")
    print(df['rank'].value_counts().sort_index())
    
    # Display first few examples
    print("\nFirst few examples:")
    print(df.head())

if __name__ == "__main__":
    main() 