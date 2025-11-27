# calculate the sentiment score using vaderSentiment textblock and flair 

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pyttsx3

nltk.download('vader_lexicon')

analyzer= SentimentIntensityAnalyzer()

# text_to_analyze="The food was fantastic, but the service was slow and frustrating."
text_to_analyze= text_lines = """I absolutely love this new phone; the camera quality is stunning!
The weather is gloomy today, but at least it’s not raining.
I can’t believe how terrible that customer service was.
This coffee tastes amazing and gives me so much energy!
The movie was okay, not great but not awful either.
I hate being stuck in traffic for hours.
What a beautiful day to take a walk in the park.
The package arrived late, and the box was damaged.
I feel so happy when I listen to my favorite song.
The restaurant food was bland and overpriced.
My dog is the cutest thing ever!
I’m tired of all the noise outside my apartment.
That presentation went surprisingly well.
The train was delayed again—so frustrating!
I love spending time with my family on weekends.
This keyboard feels cheap and uncomfortable to type on.
The sunset tonight is breathtakingly beautiful.
I regret buying this product; it broke after two days.
Today was peaceful and productive.
My laptop just crashed in the middle of my project.
That was the funniest joke I’ve heard all week!
I feel anxious about tomorrow’s meeting.
The service was excellent, and the staff was super friendly.
My internet keeps disconnecting, and it’s driving me crazy.
The concert was incredible; I had such a good time!
The food was cold and lacked flavor.
I’m grateful for my supportive friends.
The air conditioner stopped working on the hottest day of the year.
The new update makes the app much faster and smoother.
I feel disappointed with how this project turned out.
That was an amazing performance by the team.
I feel so tired and unmotivated today.
The cake was delicious and perfectly baked.
My phone battery dies so quickly; it’s annoying.
I had a relaxing afternoon reading a great book.
The meeting was boring and went on forever.
I’m so proud of myself for finishing that task.
The printer jammed again; I give up.
The flowers in the garden are blooming beautifully.
I can’t stand how rude that person was.
It feels nice to take a break after a long week.
The coffee machine broke, and I desperately need caffeine.
That was a brilliant idea—you really nailed it!
The room was dirty and smelled bad.
I feel calm and content right now.
My order was missing half the items.
The hike was challenging but so rewarding.
I had to wait an hour just to get my food.
The new design looks sleek and professional.
My flight got canceled at the last minute.
I’m so excited about the upcoming trip!
The movie’s ending was disappointing.
I love how organized my workspace looks now.
The sound quality of these headphones is terrible.
I feel inspired to start something new.
I’m bored out of my mind right now.
The beach was peaceful and relaxing.
I hate when people are late for meetings.
The dessert was heavenly; I could eat it all day.
The instructions were confusing and poorly written.
I feel so accomplished after completing that task.
The weather ruined our picnic plans.
That was a wonderful surprise!
The app keeps crashing every time I open it.
I love the smell of fresh rain.
My boss didn’t even notice my hard work.
The party was fun and full of laughter.
I’m frustrated with how slow this computer is.
The coffee shop’s atmosphere is cozy and inviting.
The service was slow and inattentive.
I feel optimistic about the future.
I can’t believe how rude that comment was.
The breakfast was perfect and just what I needed.
I feel sick and exhausted.
The view from the top of the mountain was spectacular.
The delivery driver left my package in the rain.
I had such a productive day at work.
The room was too noisy to concentrate.
That was the best pizza I’ve ever had!
I’m nervous about the exam tomorrow.
The new software update fixed all my problems.
I hate when plans get canceled at the last second.
I’m feeling confident about my goals.
The car broke down on the highway.
The teacher was kind and very helpful.
The queue was so long that I left.
I’m thrilled to see my friends again.
The product didn’t work as advertised.
The weather is perfect for a picnic.
I’m disappointed with the team’s effort today.
The show exceeded all my expectations.
I’m so tired of repeating the same mistakes.
The park was full of cheerful people.
The phone’s screen cracked after a small drop.
I feel hopeful and energized about my plans.
The traffic this morning was unbearable.
That was such a heartwarming story.
I’m annoyed by all the unnecessary emails.
The dinner was delightful and the service excellent.
I feel grateful for another peaceful day."""


sentiment_score= analyzer.polarity_scores(text_to_analyze)

print(f"\n\n\nThe text is : {text_to_analyze}")
print(f"\nThe sentiment score using Vader is : {sentiment_score}")

compound_score= sentiment_score['compound']

if compound_score >= 0.05:
    sentiment = 'Positive'
    # print("The sentiment is Positive")
    # self.speak_text("The overall sentiment is Positive")

elif compound_score <= -0.05:
    sentiment = 'Negative'
    # print("The sentiment is Negative")
    # self.speak_text("The overall sentiment is Negative")
else:
    sentiment = 'Neutral'
    # print("The sentiment is Neutral")
    # self.speak_text("The overall sentiment is Neutral")

print(f"\nOverall Sentiment (based on Compound Score): **{sentiment}**")
print(f"Compound Score: {compound_score}")
