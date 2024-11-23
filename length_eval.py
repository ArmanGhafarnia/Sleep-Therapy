def length_checker(conversation):
    # Define thresholds
    min_words = 5500
    max_words = 6500
    min_chars = 30250
    max_chars = 35750

    # Initialize counts
    total_word_count = 0
    total_char_count = 0

    # Iterate through conversation to calculate word and character counts
    for patient_sentence, therapist_sentence in conversation:
        # Count words and characters for patient
        patient_words = len(patient_sentence.split())
        patient_chars = len(patient_sentence)

        # Count words and characters for therapist
        therapist_words = len(therapist_sentence.split())
        therapist_chars = len(therapist_sentence)

        # Update total counts
        total_word_count += patient_words + therapist_words
        total_char_count += patient_chars + therapist_chars

    # Check word count range
    if min_words <= total_word_count <= max_words:
        word_check = "Pass"
    elif total_word_count < min_words:
        word_check = "Too Short"
    else:
        word_check = "Too Long"

    # Check character count range
    if min_chars <= total_char_count <= max_chars:
        char_check = "Pass"
    elif total_char_count < min_chars:
        char_check = "Too Short"
    else:
        char_check = "Too Long"

    # Return results
    return {
        "Total Word Count": total_word_count,
        "Total Character Count": total_char_count,
        "Word Check": word_check,
        "Character Check": char_check
    }

# Example conversation
conversation = [
    ("I have been feeling very anxious lately, especially at night.", "Can you tell me more about what happens when you try to sleep?"),
    ("I just can't seem to relax. I keep worrying about work and other things.", "It sounds like your mind is very active at night. Let's explore some ways to help calm your thoughts."),
    ("Sometimes I just lay in bed for hours, unable to fall asleep.", "That sounds really tough. Have you tried any relaxation techniques before bed, like deep breathing or progressive muscle relaxation?"),
    ("I've tried deep breathing, but it doesn't always work.", "That's understandable. Deep breathing can be helpful, but it's not always enough on its own. Let's talk about creating a bedtime routine that might help signal your body that it's time to sleep."),
    ("What kind of routine do you suggest?", "A good bedtime routine could include activities that help you wind down, like reading a book, taking a warm bath, or listening to calming music. The key is to do the same things each night so your body learns to associate them with sleep."),
    ("I usually watch TV before bed. Is that okay?", "Watching TV can sometimes be stimulating, especially if the content is engaging or the light is too bright. It might be helpful to switch to something more calming, like listening to a podcast or reading."),
    ("I see. I guess I could try reading instead.", "That sounds like a good idea. It might also help to set a consistent bedtime and wake-up time, even on weekends, to help regulate your sleep cycle."),
    ("I often stay up late on weekends. Could that be affecting my sleep?", "Yes, it could be. Irregular sleep schedules can confuse your body's internal clock. Sticking to a consistent schedule can make it easier to fall asleep and wake up."),
    ("I'll try to be more consistent with my sleep times.", "Great! It might take some time for your body to adjust, but consistency is key. Keeping a sleep diary can also help you track what's working and identify any patterns."),
    ("A sleep diary? How does that work?", "A sleep diary is just a simple way to record your sleep habits. You can write down when you go to bed, when you wake up, how long it takes to fall asleep, and any other factors that might be affecting your sleep. It can help us identify any changes that are helping or hindering your progress."),
    ("Okay, I'll give that a try.", "That sounds great. Remember, improving sleep can take time, but with these small changes, you'll likely start to see improvements. Let's check in next time to see how it's going."),
    ("Thank you. I feel a bit more hopeful now.", "I'm really glad to hear that. You're taking important steps, and I'm here to support you along the way."),
    ("Sometimes I wake up in the middle of the night and can't fall back asleep.", "Waking up in the middle of the night is quite common, especially when you're feeling anxious. When that happens, try not to look at the clock, and instead focus on something calming, like deep breathing or a relaxing image in your mind."),
    ("I usually end up checking my phone, which probably makes it worse.", "Yes, the light from your phone can signal your brain that it's time to wake up. It might help to keep your phone out of reach and use an alarm clock instead. That way, you won't be tempted to check it during the night."),
    ("I'll try that. It makes sense.", "Good plan. It might also be helpful to practice some mindfulness techniques during the day. The more we can calm the mind during the day, the easier it will be to settle at night."),
    ("What kind of mindfulness techniques would you recommend?", "You could try something simple like focusing on your breath for a few minutes, or doing a body scan where you pay attention to different parts of your body and release any tension you find. These techniques can help train your mind to be more present and less caught up in anxious thoughts."),
    ("I think I've heard of the body scan before. I'll give it a try.", "That's great. It's a very effective way to help relax both your body and mind. You might also consider using a guided meditation app to help you get started."),
    ("I sometimes feel like I'm not tired even though I haven't slept well.", "That can happen if your sleep cycle is out of sync or if you're running on adrenaline due to stress. It might help to avoid caffeine later in the day and to get some sunlight in the morning to help reset your internal clock."),
    ("I do drink a lot of coffee, especially in the afternoons.", "Caffeine can stay in your system for several hours, so it's best to limit it after lunchtime. Try switching to herbal tea or water in the afternoon, and see if that makes a difference."),
    ("I didn't realize caffeine could have such a long effect.", "It's true, and everyone's sensitivity to caffeine is different. Reducing your intake might help you feel more naturally tired in the evenings, which can make falling asleep easier."),
    ("I'll start cutting back and see how it goes.", "That's a good approach. Remember, these changes can take some time, but staying consistent will help your body adjust. Let's keep track of how you're feeling and discuss it during our next session."),
    ("What should I do if I still can't sleep after trying all these things?", "If you're still struggling, we could explore other options like cognitive behavioral therapy for insomnia (CBT-I). It's a structured program that helps you address the thoughts and behaviors that are keeping you from sleeping well. But for now, let's focus on the basics and see how these adjustments help."),
    ("I've heard of CBT-I. Is it really effective?", "Yes, it's actually considered the gold standard for treating insomnia without medication. It helps you reframe negative thoughts about sleep and develop habits that promote restful sleep. We can discuss it more if you'd like to explore that route in the future."),
    ("Okay, I think I'll start with these smaller changes and see if they help first.", "That sounds like a very reasonable plan. We'll take it step by step, and I'm here to support you through the process. Remember, even small improvements can be a big step in the right direction."),
    ("Thanks for all the suggestions. I feel like I have a clearer plan now.", "You're very welcome. I'm glad to hear that. Let's touch base next week to see how things are going and make any adjustments if needed."),
    ("Alright, I'll keep track of everything and let you know.", "That sounds perfect. Take care, and remember that progress takes time. You're doing great by taking these steps."),
    ("Thanks again, I appreciate it.", "You're welcome. Have a good week, and I'll look forward to hearing how things go."),
    ("I wanted to share that I tried the sleep diary and it really helped me see some patterns.", "That's wonderful to hear. Identifying patterns is a big step in understanding what's affecting your sleep. Did you notice anything specific that we can address together?"),
    ("I noticed that on days when I have a lot of caffeine, I tend to have more trouble sleeping.", "That's an important observation. Cutting back on caffeine, especially in the afternoons, can make a significant difference. It sounds like you're already starting to make connections that will help you move forward."),
    ("I also realized that on nights when I feel more relaxed before bed, I sleep better.", "Exactly. Relaxation before bed can make a big difference. Let's build on that by adding more activities that help you wind down, like listening to calming music or practicing gentle stretches."),
    ("I think listening to calming music could help. I'll try that.", "That sounds like a great idea. Music can be very soothing and help signal your body that it's time to rest. Consistency is key, so try to make it a regular part of your bedtime routine."),
    ("I've also been trying to get more sunlight in the mornings like you suggested.", "That's excellent. Morning sunlight can help regulate your circadian rhythm, making it easier to feel awake during the day and sleepy at night. Keep it up, and let's see how it continues to impact your sleep."),
    ("One thing I'm still struggling with is waking up in the middle of the night.", "Waking up during the night can be frustrating, but it's something we can work on. When you wake up, try to avoid turning on bright lights or checking your phone. Instead, focus on deep breathing or visualize something calming."),
    ("I did try the deep breathing, and it helped me relax, but sometimes it takes a while.", "That's completely normal. It can take time for your body to respond, especially if you're feeling tense. The key is to stay patient and keep practicing. Over time, it will become more effective."),
    ("I'll keep practicing. I think I'm starting to understand what works for me.", "That's great progress. Finding what works for you is a journey, and you're making important steps along the way. Let's continue building on these positive changes."),
    ("I really appreciate all your support. It feels good to have a plan.", "You're very welcome. I'm here to support you every step of the way. Having a plan can make a big difference, and you're doing a wonderful job of following through."),
    ("Thank you. I'll keep working on the routine and the diary.", "That sounds perfect. Keep up the good work, and we'll review your progress next time. Remember, small consistent efforts can lead to big changes over time."),
    ("I've also been trying to avoid caffeine, like we talked about, and I think it's helping.", "That's wonderful to hear. Reducing caffeine can definitely make a big difference in your sleep quality. I'm glad you're noticing positive changes."),
    ("Sometimes I still find it hard to wind down, though. Are there any other techniques I could try?", "Absolutely. In addition to what we've discussed, you might consider practicing yoga or gentle stretching before bed. It can help release physical tension and prepare your body for rest."),
    ("I haven't tried yoga before, but I'm open to it.", "That's great! You don't need to do anything complicated. Even just a few simple stretches can help. There are also plenty of beginner-friendly videos online that you could follow along with."),
    ("I'll look into that. I think it could be helpful.", "I'm glad to hear that. Remember, the goal is to create a routine that helps signal your body that it's time to rest. Yoga can be a great addition to that routine."),
    ("I've also noticed that when I write down my thoughts before bed, it helps me feel less anxious.", "That's an excellent strategy. Journaling can be a very effective way to clear your mind and reduce anxiety before sleep. It helps to get those thoughts out of your head and onto paper."),
    ("Sometimes I worry that I'm not making progress fast enough.", "It's completely normal to feel that way, but remember that progress is often gradual. The fact that you're making changes and noticing even small improvements is a big step forward."),
    ("I guess you're right. I do feel a bit better overall.", "That's wonderful to hear. Celebrate those small victories. Each positive change brings you closer to better sleep."),
    ("I'll keep working on it. Thank you for all your support.", "You're very welcome. You're doing a fantastic job, and I'm here to support you every step of the way. Keep up the great work.")
]

# Run the length checker
evaluation_result = length_checker(conversation)
print(evaluation_result)

