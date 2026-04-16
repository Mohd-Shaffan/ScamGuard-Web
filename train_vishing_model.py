"""
=============================================================================
  ScamGuard — Vishing Dataset Builder & Model Trainer
  Authors: Mohd Shaffan, Aditya Anurag Acharya, Shaqueeb Jamil
  Manipal University Jaipur

  This script:
    1. Loads the existing vishing_data.csv
    2. Cleans it (removes bad rows, fixes mislabels, deduplicates)
    3. Adds 400+ realistic synthetic sentences (safe + scam)
    4. Balances to equal class sizes
    5. Encodes with DistilBERT sentence embeddings (768-dim)
    6. Trains Logistic Regression with train/test split
    7. Reports accuracy, F1, confusion matrix, cross-validation
    8. Saves final model as logistic_vishing_model.pkl

  Usage:
    python train_vishing_model.py
=============================================================================
"""

import os
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, "vishing_data.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "logistic_vishing_model.pkl")
ENCODER    = "paraphrase-multilingual-MiniLM-L12-v2"   # 384-dim, native multilingual

# =============================================================================
#  STEP 1: SYNTHETIC DATA (safe + scam sentences for augmentation)
# =============================================================================
SAFE_SENTENCES = [
    # --- Casual conversation ---
    "Hey, how are you doing today?",
    "I'm doing great, thanks for asking!",
    "Are we still meeting for coffee tomorrow morning?",
    "Yes, let's meet at the usual place around 10 AM.",
    "Can you pick up some groceries on your way home?",
    "Sure, I'll stop by the store after work.",
    "Did you watch the cricket match last night?",
    "India played really well in yesterday's match.",
    "What are you planning to cook for dinner tonight?",
    "I was thinking of making chicken biryani tonight.",
    "Happy birthday! Wishing you all the best.",
    "Thank you so much! That means a lot to me.",
    "The weather is really nice today, isn't it?",
    "Let's go for a walk in the park this evening.",
    "I'll be there in about 10 minutes, wait for me.",
    "No problem, take your time, I'm not in a hurry.",
    "Did you finish the assignment that was due today?",
    "I submitted it in the morning, it was quite easy.",
    "My flight is at 6 PM, I need to leave by 3.",
    "Safe travels! Let me know when you land.",
    "Can you send me the meeting agenda for tomorrow?",
    "I'll forward the email with all the details.",
    "The new restaurant downtown is really good.",
    "We should try it this weekend for lunch.",
    "How was your day at work today?",
    "It was productive, got a lot of things done.",
    "Please remind me to call the doctor tomorrow.",
    "Sure, I'll remind you first thing in the morning.",
    "What time does the movie start tonight?",
    "The show begins at 7:30 PM at PVR cinema.",
    # --- Professional / Work ---
    "Please review the document I sent earlier.",
    "I've already reviewed it and added my comments.",
    "The project deadline has been extended by a week.",
    "That's great news, we'll have more time to prepare.",
    "Can you join the team meeting at 3 PM today?",
    "I'll be there, just finishing up a task right now.",
    "The client has approved the final design mockup.",
    "Excellent, let's start the development phase.",
    "Your leave application has been approved by HR.",
    "Thank you, I'll be away from Monday to Wednesday.",
    "The quarterly report looks impressive this time.",
    "The team has worked really hard this quarter.",
    "Please complete the training module by Friday.",
    "I'm halfway through it, will finish by Thursday.",
    "The new software update is ready for deployment.",
    "Let's schedule it for this weekend during off hours.",
    "We need to hire two more developers for the project.",
    "I'll post the job listings on LinkedIn today.",
    "The conference call with the US team is at 9 PM IST.",
    "I'll dial in from home, can you share the link?",
    # --- Family and Social ---
    "Mom, I'll be home for Diwali this year.",
    "That's wonderful! Everyone will be so happy.",
    "Congratulations on your promotion! Well deserved.",
    "Thank you, the whole team has been very supportive.",
    "The kids have their school exam next week.",
    "I'll help them study this weekend.",
    "Your cousin's wedding invitation arrived today.",
    "Great, I need to book the train tickets soon.",
    "The baby took her first steps today!",
    "Oh my God, that's amazing! Send me a video.",
    "Can you pick up the kids from school at 3 PM?",
    "I'll be there, don't worry about it.",
    "Let's plan a family trip during the summer break.",
    "How about we go to Manali or Shimla this time?",
    "Grandma is feeling much better after the surgery.",
    "That's such a relief, I'll visit her this weekend.",
    # --- Banking (safe context) ---
    "I need to visit the bank to update my passbook.",
    "The bank is open till 4 PM on weekdays.",
    "I received my salary today, the deposit is confirmed.",
    "Let me transfer the rent to the landlord now.",
    "My bank statement shows all the transactions correctly.",
    "The new debit card I ordered has arrived.",
    "I applied for a home loan at the bank branch.",
    "The loan officer said the process will take two weeks.",
    "I need to change my bank account address.",
    "You can do that through the mobile banking app.",
    "The EMI for the car loan is getting deducted automatically.",
    "That's great, auto-debit makes things so much easier.",
    "I want to open a fixed deposit for my savings.",
    "The interest rates are quite good right now.",
    "My netbanking password needs to be changed, it's been a while.",
    "Just go to settings and update it from the app.",
    # --- Technology (safe) ---
    "My phone is running low on battery, I need to charge it.",
    "There's a charger in the living room, use that.",
    "I updated my laptop to the latest operating system.",
    "How is it working? Is it faster now?",
    "The WiFi seems slow today, is something wrong?",
    "Let me restart the router, that usually fixes it.",
    "I need to back up my photos to Google Drive.",
    "Do it regularly so you don't lose any memories.",
    "The new iPhone looks really nice, thinking of buying it.",
    "Wait for a sale, you might get a good discount.",
    "Can you help me set up my new email account?",
    "Sure, which provider do you want to use?",
    # --- Education ---
    "I've enrolled in an online course on data science.",
    "That's a great field, there are many job opportunities.",
    "The professor posted the lecture notes on the portal.",
    "Thanks for letting me know, I'll download them.",
    "The campus library has some really good reference books.",
    "I'll check them out before the next assignment.",
    "Exam results will be announced tomorrow morning.",
    "I'm nervous but I studied really hard for this.",
    "The scholarship application deadline is next month.",
    "I'll prepare my documents and submit early.",
    # --- Health ---
    "I have a doctor's appointment at 4 PM today.",
    "Should I come with you for moral support?",
    "The pharmacist said to take this medicine twice a day.",
    "Make sure you eat something before taking it.",
    "I started going to the gym, feeling much healthier.",
    "That's awesome, consistency is key for fitness.",
    "The yoga class in the park is at 6 AM tomorrow.",
    "Count me in, I need some morning exercise.",
    "I'm feeling a bit under the weather today.",
    "Rest up and drink lots of water, you'll feel better.",
    # --- Shopping ---
    "I ordered a new kurta from Myntra for the party.",
    "Did you check the size? Online sizes can vary.",
    "The grocery store has a great sale this weekend.",
    "Let's make a list of what we need before going.",
    "I bought a new pair of shoes, they're really comfortable.",
    "Nice! Where did you get them from?",
    "Amazon is having a great sale on electronics.",
    "I've been wanting to buy a new headphone set.",
    # --- Travel ---
    "The flight tickets to Goa are surprisingly cheap.",
    "Book them now before the prices go up.",
    "I need to renew my passport before the trip.",
    "The passport office has an online appointment system.",
    "The hotel near the beach has great reviews.",
    "Let's book a room with an ocean view.",
    "Our train is scheduled to arrive at 8:30 AM.",
    "I'll have someone pick us up from the station.",
    # --- Food ---
    "The new cafe near the office serves amazing coffee.",
    "Let's go there for a coffee break this afternoon.",
    "I'm ordering lunch from Swiggy, want anything?",
    "Get me a paneer butter masala with naan please.",
    "Mom's special recipe for dal makhani is the best.",
    "Can you share the recipe? I want to try it.",
    # --- Warnings (safe - educating about scams) ---
    "Never share your OTP with anyone, it's your security.",
    "The bank will never call and ask for your PIN.",
    "Be careful of fake calls claiming to be from the police.",
    "Don't click on suspicious links in text messages.",
    "Always verify before transferring money to strangers.",
    "Report any fraud calls to the cyber crime helpline.",
    "Protect your personal information from scammers.",
    "Never give your Aadhaar number over the phone.",
    "Be aware of lottery scams, they are very common.",
    "If someone pressures you urgently, it's likely a scam.",
    "Don't install remote access apps like AnyDesk for strangers.",
    "Beware of KYC update scam calls, they are fake.",
    # --- Short natural responses ---
    "Sounds good, let me know the details.",
    "I completely agree with you on that.",
    "That seems like a reasonable plan to me.",
    "Let's discuss this in more detail tomorrow.",
    "I appreciate your help with this matter.",
    "Everything looks fine from my end.",
    "Good idea, let's go with that approach.",
    "Thanks for the heads up, I'll keep that in mind.",
    "No worries, these things happen sometimes.",
    "That makes perfect sense, thank you for explaining.",
    "I'm looking forward to it, should be fun.",
    "Go ahead, you have my full support.",
    "Noted, I'll make the necessary changes.",
    "That's very kind of you, thank you so much.",
    "I'll think about it and get back to you.",
    # --- More conversational ---
    "The traffic on MG Road is terrible during rush hour.",
    "Take the metro, it's much faster and cheaper.",
    "I'm watching the new series on Netflix, it's amazing.",
    "Don't spoil it for me, I haven't started yet!",
    "The street food in Delhi is absolutely incredible.",
    "Nothing beats a plate of chaat from Chandni Chowk.",
    "My neighbor's dog is so friendly, he always greets me.",
    "Dogs are the best, they make everyone smile.",
    "The monsoon season should be starting soon.",
    "I love the rain, everything looks so green and fresh.",
    "I forgot my umbrella at home and it started raining.",
    "You can borrow mine, I have an extra one.",
    "The electricity bill this month is quite high.",
    "We should switch to LED bulbs to save energy.",
    "I need to get my car serviced next week.",
    "There's a good mechanic near the main market.",

    # --- Hinglish: Casual Greetings & Small Talk ---
    "Kya haal hai bhai? Sab theek?",
    "Arey yaar, bahut din ho gaye tujhse mile hue.",
    "Kaise ho? Main theek hu, tu bata.",
    "Sab badhiya chal raha hai, tension mat le.",
    "Bhai long time no see, kab milte hain?",
    "Aaj mood kaafi acha hai, mausam bhi mast hai.",
    "Yaar tu kitna change ho gaya hai, pehchaan hi nahi aaya.",
    "Kal raat bahut maza aaya party mein.",
    "Tum bahut acche lag rahe ho aaj.",
    "Chal na baithte hain kahin, coffee peete hain.",
    "Kya kar raha hai aaj kal? Kuch naya?",
    "Main abhi ghar pahuncha, thoda rest karta hu.",
    "Tu tension mat le, sab sort ho jayega.",
    "Bhai masti karte hain weekend pe.",
    "Bahut boring din tha aaj office mein.",
    "Aaj ka din bahut productive raha mere liye.",
    "Subah se kaam mein laga hua hu, abhi free hua.",
    "Tu bhi aaja, hum sab yahi pe hain.",
    "Main thoda late ho jaunga, tu chalu kar.",
    "Haan bhai bol, kya scene hai?",
    "Arey chill kar, itna stress mat le.",
    "Tera phone nahi lag raha tha kal se.",
    "Battery dead ho gayi thi, isliye nahi utha paya.",
    "Sab log pooch rahe the tere baare mein.",
    "Bahut miss kar rahe hain sab tujhe.",
    "Kal milte hain pakka, cancel mat karna.",
    "Main free hu abhi, baat kar sakte hain.",
    "Chal theek hai, baad mein baat karte hain.",
    "Raat ko call karna, abhi meeting mein hu.",
    "Haan bhai sun, ek kaam tha tujhse.",

    # --- Hinglish: Family & Home ---
    "Mummy ne poori banayi hai aaj, jaldi aa.",
    "Papa ka BP check karwana hai kal.",
    "Bhai ki shaadi ki shopping karni hai.",
    "Ghar mein sab theek hai, chinta mat kar.",
    "Dadi ko hospital le jaana hai checkup ke liye.",
    "Mummy ne kaha tujhe ghar aake khana kha le.",
    "Bhai ka result aa gaya, pass ho gaya.",
    "Papa ne nayi gaadi li hai, bahut acchi hai.",
    "Ghar ki safai karni hai weekend pe.",
    "Bacha so gaya hai, awaaz mat karna.",
    "Didi ki job lag gayi hai Bangalore mein.",
    "Choti behen ka project help karna hai.",
    "Ghar pe mehman aa rahe hain kal.",
    "Mummy beemar hai, dawai lani hai.",
    "Papa ne kaha market se sabzi le aana.",
    "Bhabhi ne bahut accha halwa banaya hai.",
    "Nani ka phone aaya tha, sab theek hai.",
    "Ghar ki rent ka time aa gaya hai.",
    "Bijli ka bill bahut zyada aaya hai is baar.",
    "AC ka filter saaf karna hai.",
    "Washing machine repair karwani hai.",
    "Fridge mein kuch nahi hai, grocery laani padegi.",
    "Bacche school se aa gaye hain.",
    "Tuition ki fees bhar di maine.",
    "Papa retire ho rahe hain next month.",
    "Joint family mein rehna mushkil hai kabhi kabhi.",
    "Mummy ne pooch raha ki shaadi kab karoge.",
    "Ghar ka loan almost khatam hone wala hai.",
    "Diwali pe ghar whitewash karwana hai.",
    "Garden mein naye paudhe lagaye hain.",

    # --- Hinglish: Office & Work ---
    "Aaj office mein bahut kaam tha.",
    "Boss ne naya project assign kiya hai.",
    "Meeting 3 baje hai, yaad rakhna.",
    "Salary aa gayi kya tere account mein?",
    "Leave apply karni hai kal ke liye.",
    "Appraisal mein kuch khaas nahi mila is baar.",
    "Naya intern aaya hai, bahut smart hai.",
    "Office ki canteen ka khana aaj accha tha.",
    "Deadline extend ho gayi hai ek hafte se.",
    "Presentation ready kar li maine.",
    "Client ko mail bhej dena aaj shaam tak.",
    "Code review pending hai tera, jaldi kar.",
    "Laptop bahut slow ho gaya hai, IT se bolna padega.",
    "Kaam khatam kare bina ghar nahi ja sakta aaj.",
    "Office shift timing change ho gayi hai.",
    "Work from home ka option mil gaya hai.",
    "Team outing plan ho rahi hai next week.",
    "HR se baat karni hai leave ke baare mein.",
    "Monthly report submit karni hai kal tak.",
    "Server down hai, kuch kaam nahi ho pa raha.",
    "Git merge conflict aa raha hai, help chahiye.",
    "Standup meeting skip kar di aaj maine.",
    "Tera code accha hai bhai, clean likha hai.",
    "Production deploy karna hai raat ko.",
    "Sprint planning kal hai, backlog update kar.",
    "Manager ne appreciation mail bheja hai.",
    "Office mein WiFi bahut slow hai aaj.",
    "Cab late aa rahi hai roz subah.",
    "Cafeteria mein naya menu aaya hai.",
    "Conference room book kar de 4 baje ka.",

    # --- Hinglish: Food & Cooking ---
    "Aaj kya khayein? Bahar chalte hain kya?",
    "Biryani khane ka mann hai.",
    "Chole bhature wala naya dukaan khula hai.",
    "Mummy ki haath ki dal baat alag hai.",
    "Zomato se order karte hain kuch.",
    "Swiggy pe 50 percent off hai aaj.",
    "Chai bana de yaar, bahut thak gaya hu.",
    "Maggi khani hai raat ko.",
    "Paneer tikka banana seekh raha hu.",
    "Atta khatam ho gaya hai, laana padega.",
    "Sabzi mandi se taaza sabziya layi hain.",
    "Momos khane chalein? Naye stall pe.",
    "Chutnee bahut acchi banayi hai tune.",
    "Roti gol nahi ban rahi meri.",
    "Doodh le aana dairy se.",
    "Fruit salad kha le, healthy rehna chahiye.",
    "Pizza order karte hain Dominos se.",
    "Paratha accha bana hai aaj tera.",
    "Kheer bana rahi hai mummy Navratri ke liye.",
    "Samosa garam garam kha, bahut mast hai.",
    "Restaurant ka khana itna mehenga ho gaya hai.",
    "Ghar ka khana hi sabse acha hai.",
    "Filter coffee try kar ek baar, mast hai.",
    "Ice cream kha le, garmi bahut hai.",
    "Nimbu paani pee le, dehydration ho jayega.",

    # --- Hinglish: Education & Exams ---
    "Exam ki padhai shuru kar di?",
    "Notes share karna apne, mere pass nahi hai.",
    "Viva kal hai, bahut nervous hu.",
    "Attendance kam ho gayi hai, dhyan rakh.",
    "Professor ne assignment diya hai Monday tak.",
    "Library mein seat nahi milti aaj kal.",
    "Placement season shuru hone wala hai.",
    "GATE ki taiyari kar raha hu.",
    "CAT ke liye coaching join karni hai.",
    "College fest next month hai, volunteer karega?",
    "Lab practical ka submission kal hai.",
    "Semester result aa gaya, CGPA 8.5 hai.",
    "Internship mil gayi, Amazon mein.",
    "Thesis complete karni hai March tak.",
    "Research paper publish karwana hai.",
    "Online class boring hoti hai yaar.",
    "Group discussion ki practice karni hai.",
    "Hostel ka khana kab sudharega.",
    "Canteen mein samosa bahut mehenga ho gaya.",
    "Farewell party ka plan banana hai.",
    "Junior ko ragging nahi karni chahiye.",
    "Mentor bahut helpful hai apna.",
    "Dean ne notice diya hai attendance ke liye.",
    "Workshop attend karni hai Saturday ko.",
    "Coding competition mein participate karna hai.",

    # --- Hinglish: Shopping & Market ---
    "Mall mein sale laga hai, chalein kya?",
    "Amazon pe bahut acche deals hain aaj.",
    "Flipkart Big Billion Day aane wala hai.",
    "Naye kapde lene hain Diwali ke liye.",
    "Shoes purane ho gaye hain, naye chahiye.",
    "Ye shirt tujh pe acchi lagegi.",
    "Return karna hai ye product, size galat aaya.",
    "Online shopping mein fraud se bachna chahiye.",
    "Local market mein sasta milega ye.",
    "Meesho pe bahut sasti cheezein milti hain.",
    "Myntra se order kiya tha, aa gaya.",
    "Coupon code lagaya toh 200 bach gaye.",
    "EMI pe le le, ek saath nahi dene padenge.",
    "Sabzi mandi mein tamatar 80 rupaye kilo hai.",
    "Petrol phir mehenga ho gaya hai.",
    "Groceries ka kharcha bahut badh gaya hai.",
    "Ye phone lena chahiye, reviews acche hain.",
    "Warranty khatam ho gayi kya iska?",
    "Bill dikha de, exchange karna hai.",
    "Chottu ki dukaan pe sab sasta milta hai.",

    # --- Hinglish: Travel & Transport ---
    "Goa ka trip plan karte hain.",
    "Flight ki tickets bahut mehengi hain abhi.",
    "Train mein confirm ticket mil gayi.",
    "Ola book kar, auto nahi mil raha.",
    "Uber mein surge pricing hai abhi.",
    "Metro se jao, traffic bahut hai.",
    "Road pe bahut traffic jam hai aaj.",
    "Petrol bharwa le, tank khaali hai.",
    "Manali mein bahut thand hai abhi.",
    "Kashmir jaana hai is winter mein.",
    "Passport renew karwana hai.",
    "Visa appointment leni hai US ki.",
    "Airport pe 2 ghante pehle pahunchna padega.",
    "Hotel ka booking kar di hai Jaipur mein.",
    "Rajasthan mein desert safari karni hai.",
    "Kerala trip pe bahut maza aaya tha.",
    "Ladakh bike trip ka plan hai.",
    "IRCTC pe tatkal booking karni hai.",
    "Bus mein seat reserve karwa li.",
    "Challan kat gaya signal todne ka.",

    # --- Hinglish: Sports & Entertainment ---
    "IPL match dekh raha hai? RCB vs CSK hai aaj.",
    "Virat ne century maari, kya innings tha!",
    "India ne World Cup jeet liya!",
    "FIFA World Cup final mein kya match tha.",
    "Cricket practice ke liye ground pe jaana hai.",
    "Gym join karna hai, health pe dhyan dena chahiye.",
    "Yoga karna shuru kiya hai subah.",
    "Running karta hai tu roz? Main bhi start karunga.",
    "Movie dekhne chalein? Naya release aaya hai.",
    "OTT pe bahut acchi series aayi hai.",
    "Netflix pe kya dekh raha hai aaj kal?",
    "Amazon Prime pe naya show aaya hai.",
    "Spotify pe ye song sun, bahut accha hai.",
    "Concert ka ticket milega kya?",
    "Gaming karte hain raat ko, BGMI khelenge.",
    "PS5 bahut mehenga hai yaar.",
    "Instagram pe tera reel viral ho gaya.",
    "YouTube pe vlog upload karna hai.",
    "Podcast sunna shuru kiya hai maine.",
    "Book padh raha hu ek, bahut interesting hai.",

    # --- Hinglish: Health & Wellness ---
    "Tabiyat theek nahi hai, bukhar aa raha hai.",
    "Doctor ke paas jaana hai kal.",
    "Dawai kha li? Time pe leni chahiye.",
    "Health insurance le le, zaruri hai.",
    "Blood test karwana hai routine checkup ka.",
    "Dentist ka appointment lena hai.",
    "Neend nahi aa rahi raat ko.",
    "Stress bahut badh gaya hai kaam ka.",
    "Walking kar, health acchi rahegi.",
    "Pani zyada piya kar, dehydration hota hai.",
    "Aankhein dard kar rahi hain screen dekhte dekhte.",
    "Physiotherapy leni padegi back pain ke liye.",
    "Vitamins le raha hu daily.",
    "Weight badh gaya hai, diet pe aana padega.",
    "Allergy ho gayi hai kisi cheez se.",
    "COVID vaccine laga li? Booster bhi lagwa le.",
    "Mental health ka bhi khayal rakhna chahiye.",
    "Meditation try kar, bahut fayda hota hai.",
    "Gym mein injury ho gayi, rest karna padega.",
    "Protein shake peeta hai tu workout ke baad?",

    # --- Hinglish: Technology ---
    "Phone ka storage full ho gaya.",
    "Laptop mein RAM upgrade karwana chahiye.",
    "WiFi bahut slow chal raha hai ghar ka.",
    "Naya iPhone launch hua hai, dekha tune?",
    "App update kar le, naya version aaya hai.",
    "UPI se payment kar de, cash nahi hai.",
    "Google Drive pe file upload kar de.",
    "Instagram story dal di maine.",
    "WhatsApp group mein message dekh.",
    "LinkedIn pe profile update karna hai.",
    "Naya laptop lena hai coding ke liye.",
    "Screen guard lagwa le phone pe.",
    "Charger kharab ho gaya, naya lena padega.",
    "Bluetooth speaker kharid liya maine.",
    "Printer mein ink khatam ho gayi.",
    "Windows update install karna hai.",
    "Antivirus renew karwana padega.",
    "Cloud backup le le data ka.",
    "VPN use kar public WiFi pe.",
    "Dark mode on kar le, aankhon ko aaram milta hai.",

    # --- Hinglish: Festivals & Celebrations ---
    "Diwali ki shopping shuru karte hain.",
    "Holi pe rang nahi khelna mujhe.",
    "Eid pe biryani khane aaja ghar.",
    "Christmas party rakh rahe hain office mein.",
    "Navratri mein garba khelne chalein?",
    "Raksha Bandhan pe behen ko gift dena hai.",
    "Ganesh Chaturthi pe pandal dekhne jayenge.",
    "Makar Sankranti pe patang udaayenge.",
    "Durga Puja mein pandal hopping karni hai.",
    "New Year party kahan pe hai?",
    "Birthday cake order karna hai.",
    "Anniversary gift kya dein?",
    "Shaadi ka card aaya hai, jaana padega.",
    "Mehendi lagwani hai engagement ke liye.",
    "Sangeet mein dance practice karni hai.",

    # --- Hinglish: Weather & Seasons ---
    "Baarish ho rahi hai bahar, chata le ja.",
    "Bahut garmi hai aaj, AC chala de.",
    "Thand bahut hai, sweater pehen le.",
    "Fog itni hai ki kuch dikh nahi raha.",
    "Dhoop mein mat nikalna, sunstroke ho jayega.",
    "Mausam bahut suhana hai aaj.",
    "Barsat mein chai pakode ka maza alag hai.",
    "Summer vacation ka plan bana.",
    "Winter mein Shimla jaayenge.",
    "Spring mein garden kitna accha lagta hai.",

    # --- Hinglish: Banking & Finance (SAFE context) ---
    "Salary account mein paise aa gaye.",
    "FD karwa li hai 7 percent interest pe.",
    "Mutual fund mein SIP shuru ki hai.",
    "Income tax return file karni hai.",
    "Credit score check kiya, 750 hai.",
    "EPF ka balance dekh le online.",
    "Insurance premium bharna hai is month.",
    "Home loan ka EMI auto debit hai.",
    "UPI se dost ko paise bhej diye.",
    "Paytm wallet mein balance add karna hai.",
    "NEFT se payment bhej di hai.",
    "ITR filing last date 31st July hai.",
    "Stock market mein invest karna seekh raha hu.",
    "PPF account mein deposit karna hai.",
    "Budget manage karna seekhna chahiye.",

    # --- Hinglish: Positive Warnings / Advisory (SAFE) ---
    "OTP kisi ko mat batana, fraud ho sakta hai.",
    "Online scam se bachne ke liye aware raho.",
    "Unknown links pe click mat karna.",
    "Bank kabhi phone pe OTP nahi maangta.",
    "Aadhaar number share mat karo kisi se.",
    "Public WiFi pe banking mat karna.",
    "Strong password rakhna bahut zaruri hai.",
    "Two factor authentication enable kar le.",
    "Suspicious call aaye toh disconnect kar de.",
    "Phishing email ka reply mat karna.",
    "Apna PIN kisi ko mat batao, ye tumhari security hai.",
    "Cyber crime helpline 1930 pe call kar sakte ho.",
    "RBI ne kaha hai ki bank kabhi personal details nahi maangta.",
    "Fake calls se savdhan raho, pehle verify karo.",
    "Net banking ka password regularly change karo.",
]

SCAM_SENTENCES = [
    # --- Bank account scams ---
    "Hello sir, your SBI bank account has been blocked due to suspicious activity. Please share your OTP to unblock it.",
    "This is an urgent call from HDFC bank. Your account will be permanently closed if you don't verify your details now.",
    "We have detected unauthorized transactions on your ICICI account. Provide your CVV and expiry date to secure it.",
    "Your bank account has been compromised. Transfer all funds to this safe account immediately.",
    "RBI has flagged your account for illegal transactions. Share your net banking password to verify.",
    "Your debit card has been blocked. Press 1 and provide your card details to reactivate.",
    "This is Axis Bank customer care. Your account has unusual activity. Share your UPI PIN for verification.",
    "Your SBI YONO account will be suspended in 24 hours. Click this link to update your KYC immediately.",
    "Alert from HDFC Bank: Your account is under investigation. Call this number with your account details.",
    "Your bank account has been selected for a security upgrade. Share your ATM card number and PIN.",
    # --- KYC scams ---
    "Your KYC has expired. Your Paytm wallet will be blocked if not updated within 2 hours.",
    "Urgent: Update your KYC by sharing your Aadhaar number and PAN card to avoid account suspension.",
    "PhonePe requires mandatory KYC update. Share your documents on WhatsApp to this number.",
    "Google Pay KYC verification needed. Download this app and scan your Aadhaar for instant verification.",
    "Your GPay account will be deactivated without KYC. Share your PAN and Aadhaar now.",
    # --- Police/Legal scams ---
    "This is the cyber crime department. A case has been filed against your Aadhaar number. Pay the fine immediately.",
    "Your name is linked to a money laundering case. Transfer the security deposit or face arrest.",
    "FIR has been registered against your mobile number. Cooperate by sharing your details to avoid arrest.",
    "This is the Income Tax department. You have pending tax dues. Pay now to avoid legal action.",
    "A drug parcel with your name was intercepted by customs. Pay the clearance fee to avoid jail time.",
    "We are calling from the Supreme Court. Your hearing is tomorrow. Pay the bond amount now.",
    "Police case number 4523 has been filed against you. Only way to dismiss is by paying the settlement.",
    "Your Aadhaar has been used in a terrorism case. Cooperate now or we will issue arrest warrant.",
    "CBI is investigating your phone number for fraud. Verify your identity by sharing Aadhaar and PAN.",
    "TRAI has received complaints about your number. It will be disconnected in 2 hours unless you verify.",
    # --- OTP/Remote access scams ---
    "I'm calling from Amazon. Your order refund is pending. Share the OTP you received to process it.",
    "An OTP has been sent to your phone by mistake. Can you please read it out to me?",
    "For refund processing, please install AnyDesk and share the 9-digit code with me.",
    "Install TeamViewer urgently. We need remote access to fix a security issue with your phone.",
    "I'll send you an OTP for account verification. Please share it with me when you receive it.",
    "Your WhatsApp will be hacked if you don't share the verification code I just sent.",
    "To complete the KYC, I need the OTP that was sent to your registered mobile number.",
    "I'm from Google support. Share your Gmail OTP to prevent your account from being deleted.",
    "Install QuickSupport app and give me remote access. I'll fix the virus on your phone.",
    "The OTP I sent is for security verification only. It is completely safe to share with me.",
    # --- Lottery/Prize scams ---
    "Congratulations! You've won Rs 25 lakh in the KBC lottery. Pay processing fee to claim.",
    "You have been selected for a special government reward of Rs 15 lakh. Share your bank details.",
    "Your mobile number has won the Jio lucky draw prize of Rs 50,000. Pay tax to receive it.",
    "You are the winner of our international lottery worth $500,000. Transfer the handling charges.",
    "Flipkart has selected you as the lucky winner. Pay Rs 5,000 registration fee to claim your prize.",
    "Amazon quiz winner! You've won an iPhone. Pay delivery charges of Rs 2,999 to receive it.",
    "Your number was randomly selected for Rs 10 lakh reward from RBI. Provide account details.",
    "Congratulations on winning the WhatsApp lucky draw. Transfer Rs 3,000 to receive Rs 1 crore.",
    # --- Loan/Credit scams ---
    "Pre-approved personal loan of Rs 10 lakh at 0% interest. Share Aadhaar to avail immediately.",
    "Your credit card limit has been increased. Share your card number for activation.",
    "Outstanding loan of Rs 2,50,000 on your account. Pay immediately or face legal consequences.",
    "We can settle your home loan at 50% discount. Transfer the settlement amount today.",
    "Your CIBIL score has dropped. Pay Rs 5,000 to restore it immediately.",
    "Low interest car loan approved for you. Just share your PAN and salary slip to process.",
    # --- UPI/Digital payment scams ---
    "I'm sending you money on UPI. Please accept the collect request.",
    "I accidentally transferred Rs 50,000 to you. Please return it to this UPI ID immediately.",
    "To receive the cashback of Rs 10,000, scan this QR code from your GPay app.",
    "Your PhonePe account has been hacked. Send Rs 1 to verify your UPI is working.",
    "Enter your UPI PIN to receive the refund of Rs 5,000 that's pending for you.",
    "Complete this UPI payment of Re 1 for verification and get Rs 1,000 cashback instantly.",
    # --- Insurance scams ---
    "Your LIC policy is about to lapse. Share your policy number and pay the premium now.",
    "You are eligible for a health insurance bonus of Rs 5 lakh. Share your details to claim.",
    "Your insurance claim has been approved. Pay the processing fee of Rs 10,000 to receive it.",
    "Special government health scheme covers you for free. Provide Aadhaar for enrollment.",
    # --- Electricity/Utility scams ---
    "Your electricity connection will be cut in 2 hours due to pending bill. Pay now.",
    "BSNL will disconnect your number today. Call immediately with your Aadhaar to prevent.",
    "Your gas connection will be terminated. Pay the outstanding amount of Rs 8,500 now.",
    "Water supply will be stopped due to unpaid bill. Transfer Rs 3,200 immediately.",
    # --- Job scams ---
    "Work from home opportunity. Earn Rs 50,000 per month. Just pay Rs 2,000 registration fee.",
    "You've been selected for a government job. Pay the security deposit of Rs 15,000.",
    "Amazon is hiring data entry operators. Share your resume and pay Rs 500 for processing.",
    "Part time job offer: Like videos and earn Rs 5,000 per day. Join now on Telegram.",
    # --- Delivery/Parcel scams ---
    "Your courier from international has been held at customs. Pay Rs 25,000 clearance fee.",
    "A valuable parcel addressed to you requires identity verification. Share Aadhaar immediately.",
    "Your FedEx shipment requires additional charges. Pay Rs 12,000 or it will be returned.",
    "Customs duty of Rs 50,000 pending on your imported goods. Pay to avoid seizure.",
    # --- Investment scams ---
    "Invest Rs 10,000 in our stock market scheme and earn Rs 1 lakh guaranteed in 30 days.",
    "Cryptocurrency trading opportunity. Double your money in just 7 days. Guaranteed returns.",
    "Join our WhatsApp group for guaranteed stock market tips. Just pay Rs 5,000 monthly.",
    "Binary trading platform with 100% profit guarantee. Invest now before the offer expires.",
    # --- SIM/Mobile scams ---
    "Your SIM will be blocked in 4 hours due to Aadhaar mismatch. Share Aadhaar to verify.",
    "Your mobile number is being used for fraud. Send Aadhaar to this number to clear your name.",
    "TRAI will deactivate your number. Press 1 to speak with an officer immediately.",
    "Your JIO SIM has been flagged. Visit the link and enter your details to avoid blocking.",
    # --- Emotional manipulation ---
    "Your son has been detained by the police. Transfer Rs 2 lakh bail money immediately.",
    "This is the hospital. Your relative has had an accident. Transfer money for emergency surgery.",
    "I am your son's professor. He is in trouble and needs Rs 50,000 immediately.",
    "Your daughter has been kidnapped. Transfer money to this account and don't call police.",
    # --- Mixed/Sophisticated ---
    "As per RBI guidelines, you need to verify your account by sharing your net banking credentials.",
    "We're from the government digital India program. Install this app for free digital services.",
    "Your pension will be stopped unless you update your biometric data. Share Aadhaar urgently.",
    "Central government scheme for free laptop. Pay Rs 1,500 processing and shipping fee only.",
    "Aadhaar biometric update required. Visit this website and upload your fingerprint data.",
    "Your PF account has matured. Share your UAN number and bank details for withdrawal.",
    "Income tax refund of Rs 25,000 pending. Enter your card details to receive it.",
    "Your Aadhaar has been deactivated. Reactivate by providing your biometric details.",
    "Government is distributing free ration kits. Share Aadhaar for registration.",
    "PM Kisan Yojana rejected your application. Verify with Aadhaar now to reapply.",
    # --- Pressure/urgency based ---
    "This is your last warning. Your account will be permanently deleted in 1 hour.",
    "Do not hang up the phone. Stay on the line or the police will be sent to your address.",
    "Act immediately or your savings will be lost. There is no time to waste.",
    "Do not tell anyone about this call. It is a confidential government matter.",
    "If you don't cooperate right now, a warrant will be issued in your name today.",
    "Transfer the amount in the next 30 minutes or the legal proceedings will begin.",
    "Keep this call confidential. Do not discuss this with family or friends.",
    "You must act now. Every minute you delay, your risk increases.",
    "This is extremely urgent. Your financial security is at stake right now.",
    "We will freeze all your accounts if you do not comply within the next hour.",

    # --- Hinglish Scam: Banking & OTP ---
    "Sir main SBI head office se baat kar raha hu. Aapka account block ho gaya hai.",
    "Aapna khaata chalu rakhne ke liye turant apna aadhaar link karein.",
    "Aapke account se 50000 rupe nikalne wale hain. Rokne ke liye OTP bataiye.",
    "Madam ji, aapka ATM card expire ho raha hai. Naya card lene ke liye details dein.",
    "KYC update nahi karaya toh kal se bank transaction band ho jayega.",
    "Aapke credit card pe fine laga hai. Maaf karwana hai toh OTP share kariye.",
    "Sir, aapko refund mila hai 5000 ka. UPI pin enter kariye paise paane ke liye.",
    "Main HDFC bank manager bol raha hu. Aapne payment nahi kiya EMI ka.",
    "Aapka loan approve ho gaya hai 5 lakh ka. File charge 2000 bhejein.",
    "Google Pay cashback aaya hai aapka. Reward link pe click kijiye.",
    "PhonePe customer care se hoon. Aapka transaction fail hua, refund ke liye verify karein.",
    "Pan card aadhaar se link nahi hai aapka, isliye account freeze ho gaya hai.",
    "Online 5000 transfer ho gaye hain, cancel karne ke liye AnyDesk app download karein.",
    "Screen share app install kariye, main aapka account wapas theek kar dunga.",
    "Sir turant apna PIN bataiye, nahi to bank band kar dega account.",
    "Agar aap chahte hain ki aapke paise safe rahein to password verify kariye.",
    "TeamViewer app se aap apna phone mujhe connect karein, main help karunga.",
    "Dear customer, aapka Yono SBI block ho gaya hai. Link par click karein.",
    "HDFC netbanking band ho jayegi agar aapne apna details abhi form me nahi bhara.",
    "ICICI credit card limit badhani hai? Apna card detail aur CVV message karein.",
    "Aapka account hack ho gaya hai. Security verify karne ke liye OTP batayein.",
    "Mera naam Rahul hai PNB bank se, aapka khata hold pe rakha gaya hai.",
    "Khata band hone wala hai 24 ghante mein. Turant call karein is number par.",
    
    # --- Hinglish Scam: Police, Fine & Legal ---
    "Main cyber crime branch se Inspector Sharma bol raha hu.",
    "Aapke khilaaf FIR darz hui hai dhokhadhadi ke mamle mein.",
    "Agar aap arrest nahi hona chahte toh 50000 rupe penalty bhariye.",
    "Aapka beta arrest ho gaya hai, usko chhudana hai toh paise bhejiye.",
    "Police aane wali hai aapke ghar 10 minute mein. Case band karwana hai toh batao.",
    "Aapne traffic signal toda tha, fine ka message gaya hai, link se pay karo.",
    "Main supreme court se advocate bol raha hu, aapke naam pe warrant hai.",
    "Jail jana padega agar aapne kal tak penalty ka paisa nahi jama kiya toh.",
    "Aapke parcel me ganja nikla hai. Customs officer isko seize kar lenge.",
    "Parcel courier custom duty nahi bhari aapne. Giraftari hogi aapki.",
    "Aapka passport cancel hone wala hai illegal activity ke liye.",
    "Giraftar hona padega sir, CID se complaint likhwai gayi hai.",
    "Pulis ne complaint likhi hai. Is case ko settle karne ke liye paise transfer karo.",
    "Main CBI se baat kar raha hu. Aapka account money laundering me pakda gaya.",
    "Aapko abhi thikane laga denge, agar chupchap payment nahi ki to.",
    "Agar aapne fine nahi bhara toh court case hoga.",
    "Ye aakhri warning hai, police station aa jaiye ya online bail lijiye.",
    "Bail paper ban gaye hain, 1 lakh jamanat bhari padegi turant.",

    # --- Hinglish Scam: Lottery, Jobs & Prizes ---
    "KBC se bol raha hu, aapki 25 lakh ki lottery lagi hai.",
    "Congratulations sir! Aap Tata Motors ke lucky draw winner hain.",
    "Inaam ki rashi lene ke liye 1500 rupe GST tax bharna padega.",
    "Aapka number select hua hai free iPhone ke liye. Courier charge de do.",
    "Part time job karna hai? Ghar baithe 2000 kamaye roz.",
    "Data entry ki job hai madam, registration fees sirf 1000 rupaye hai.",
    "Amazon Work From home me aapka selection ho gaya hai. HR se baat kariye.",
    "YouTube ke videos like karke aap monthly 50,000 kama sakte hain.",
    "Sir aapko 5 crore ka jackpot laga hai UK wali company se.",
    "Lottery amount account me lene ke liye bank ka ifsc aur pin batao.",
    "Aap lucky winner ban gaye hain Diwali offer ke tahat.",
    "Jeet gaye aap ek naya gaadi! Delivery charge jama karayein aaj hi.",
    "Refund mangna hai toh customer id aur apna passcode bhejiye.",
    "Kaun Banega Crorepati WhatsApp lottery contest me apka naam hai.",
    "Aapko ek free trip mili hai Dubai ki. Visa processing fees lagti hai.",
    "Stock market me daily double profit kamana hai toh humein paise bhejein.",
    "Bitcoin trading me 100 percent guaranteed return milega, join kariye.",

    # --- Hinglish Scam: Support & Telecom ---
    "TRAI se baat kar raha hu, aapka SIM card deactivate hone wala hai.",
    "Jio company se bol raha hu, 5G upgrade ke liye ye OTP confirm karein.",
    "Airtel number block ho jayega document incomplete hone ke wajah se.",
    "Aapka TV DTH connection disconnect hone wala hai, recharge karein link se.",
    "Aapke phone me virus aaya hai, humara app install kariye hatane ke liye.",
    "Amazon Prime free subscription chahiye toh is link par click karein.",
    "Customer care executive bol raha hu, aapki problem thik kar dunga, detail do.",
    "Sir aapka refund block hai. AnyDesk app se check karunga apka error.",
    "Aadhar card update karana hai toh is government portal me OTP daaliye.",
]

# Patterns that indicate something was mislabeled as scam (it's actually safe)
MISLABEL_PATTERNS = [
    "backed up", "snapshot created", "badge is unlocked", "share achievement",
    "revision is reverted", "canonical is set", "retention is",
    "access card is activated", "library fine", "is verified. welcome",
    "is restored. previous", "playlist is created", "language is set",
    "region is set", "currency is set", "support ticket is created",
    "issue is resolved", "recovery email is set", "dependency is resolved",
    "database is backed", "domain is verified", "build successful",
    "reminder is set. you will be notified", "data is restored",
    "complaint is resolved. rate", "your badge is ready",
    "your reminder is set", "your complaint is resolved",
    "your access card is activated", "your data is restored",
    "your issue is resolved", "your support ticket is created",
    "your recovery email is set", "your dependency is resolved",
    "your domain is verified", "your playlist is created",
    "your language is set", "your region is set", "your currency is set",
    "your badge is unlocked", "your email is verified",
]


def main():
    print("=" * 65)
    print("  SCAMGUARD — DATASET BUILD & MODEL TRAINING")
    print("  Manipal University Jaipur")
    print("=" * 65)

    # ── STEP 1: Load existing data ──────────────────────────────────────
    print("\n[1/8] Loading existing vishing_data.csv ...")
    df = pd.read_csv(CSV_PATH)
    print(f"  Loaded {len(df)} rows  (safe={len(df[df['label']==0])}, scam={len(df[df['label']==1])})")

    # ── STEP 2: Clean ───────────────────────────────────────────────────
    print("\n[2/8] Cleaning dataset ...")
    before = len(df)

    # Remove duplicates
    df = df.drop_duplicates(subset=["text"])
    print(f"  Deduplication:  {before} -> {len(df)} rows")

    # Remove single-word safe entries (useless for sentence classification)
    df = df[~((df["label"] == 0) & (~df["text"].str.contains(" ", na=False)))]

    # Remove very short safe entries (< 15 chars)
    df = df[~((df["label"] == 0) & (df["text"].str.len() < 15))]
    print(f"  After short removal:  {len(df)} rows")

    # Fix mislabeled scam entries (actually safe notifications)
    def is_mislabeled(text):
        tl = str(text).lower()
        return any(p in tl for p in MISLABEL_PATTERNS)

    mask = (df["label"] == 1) & df["text"].apply(is_mislabeled)
    mislabeled_count = mask.sum()
    df.loc[mask, "label"] = 0
    print(f"  Fixed {mislabeled_count} mislabeled scam->safe")

    # Remove scam entries that are single-word or very short
    df = df[~((df["label"] == 1) & (df["text"].str.len() < 15))]
    print(f"  After scam short removal:  {len(df)} rows")
    print(f"  Current:  safe={len(df[df['label']==0])}, scam={len(df[df['label']==1])}")

    # ── STEP 3: Add synthetic data ──────────────────────────────────────
    print("\n[3/8] Adding synthetic sentences ...")
    safe_df = pd.DataFrame({"text": SAFE_SENTENCES + [
        "Hey bhai, kal office kitne baje aana hai?", "Mummy khana tayyar hai kya?",
        "Main 10 minute mein wahan pahunch raha hu.", "Exam ki tayari kaisi chal rahi hai?",
        "Bhai movie dekhne chale aaj raat?", "Kal subah jaldi uthna hai, alarm laga de.",
        "Mera laptop theek nahi chal raha, check kar lena.", "Office ka naya project kab start hoga?",
        "Aapka din kaisa raha?", "Sab theek thak chal raha hai, tension mat lo.",
        "Sir mujhe kal chhutti chahiye.", "Meeting ka link share karna ek baar.",
        "Main driving kar raha hu, baad mein call karta hu.",
        "Paanipuri waale bhaiya theek se banana.",
        "Zomato wala aa gaya hai, main gate par hu."
    ], "label": 0})
    scam_df = pd.DataFrame({"text": SCAM_SENTENCES + [
        "Sir main sbi branch se bol raha hu, aapka khata band hone wala hai.",
        "Aapke number par kbc lottery laga hai 25 lakh ka, apna account number dein.",
        "Yeh police station se hai, aapke khilaaf FIR darz hui hai paiso ke mamle me.",
        "Madam aapka credit card block ho gaya hai, unblock karne ke liye OTP batayein.",
        "Turant anydesk install kijiye warna aapka aadhaar link fail ho jayega.",
        "Main custom officer bol raha hu, aapka parcel pakda gaya hai, fine bharna padega.",
        "Jaldi paisa transfer kariye warna police aake giraftar kar legi.",
        "Aapko EMI waiver mila hai sir, bas ek link pe click karke pin daliye.",
        "Ye antim chetawani hai, aapka khata freeze hone ja raha hai.",
        "Jaldi account update kijiye, aakhri mauka hai."
    ], "label": 1})
    df = pd.concat([df, safe_df, scam_df], ignore_index=True)
    df = df.drop_duplicates(subset=["text"])
    df = df[df["text"].str.len() >= 10]
    print(f"  Enhanced dataset:  {len(df)} rows  (safe={len(df[df['label']==0])}, scam={len(df[df['label']==1])})")

    # ── STEP 4: Balance ─────────────────────────────────────────────────
    print("\n[4/8] Balancing classes ...")
    safe_all = df[df["label"] == 0]
    scam_all = df[df["label"] == 1]
    min_count = min(len(safe_all), len(scam_all))
    use_count = min(min_count, 5000)  # cap at 5000 per class

    safe_bal = safe_all.sample(use_count, random_state=42)
    scam_bal = scam_all.sample(use_count, random_state=42)
    df_final = pd.concat([safe_bal, scam_bal]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Balanced:  {len(df_final)} rows  ({use_count} per class)")

    # ── STEP 5: Save cleaned dataset ────────────────────────────────────
    print("\n[5/8] Saving cleaned dataset ...")
    df.to_csv(CSV_PATH, index=False)
    print(f"  Saved to {CSV_PATH}  ({len(df)} total rows)")

    # ── STEP 6: Encode with DistilBERT ──────────────────────────────────
    print(f"\n[6/8] Encoding with {ENCODER} ...")
    encoder = SentenceTransformer(ENCODER)
    texts  = df_final["text"].tolist()
    labels = df_final["label"].tolist()
    embeddings = encoder.encode(texts, show_progress_bar=True, batch_size=64)
    print(f"  Embedding shape: {embeddings.shape}")

    # ── STEP 7: Train with validation ───────────────────────────────────
    print("\n[7/8] Training Logistic Regression ...")
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

    # Scale embeddings for better LR convergence
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42, class_weight="balanced")
    clf.fit(X_train_s, y_train)

    y_pred   = clf.predict(X_test_s)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\n  Test Accuracy:  {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Safe", "Scam"]))

    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"              Pred Safe  Pred Scam")
    print(f"  Act Safe      {cm[0][0]:4d}       {cm[0][1]:4d}")
    print(f"  Act Scam      {cm[1][0]:4d}       {cm[1][1]:4d}")

    # Cross-validation
    print("\n  5-Fold Cross-Validation ...")
    # Re-scale all embeddings for CV
    all_scaled = scaler.fit_transform(embeddings)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, all_scaled, labels, cv=cv, scoring="accuracy")
    print(f"  CV Scores: {np.round(cv_scores, 4)}")
    print(f"  Mean CV Accuracy: {cv_scores.mean():.4f}  (±{cv_scores.std():.4f})")

    # ── STEP 8: Train final model on ALL data & save ────────────────────
    print("\n[8/8] Training final model on full balanced dataset & saving ...")
    all_scaled = scaler.fit_transform(embeddings)
    clf_final = LogisticRegression(max_iter=2000, C=1.0, random_state=42, class_weight="balanced")
    clf_final.fit(all_scaled, labels)

    # NOTE: The deployed model doesn't use scaler (scamguard_enhanced.py calls
    # encoder.encode + clf.predict_proba directly). So we save an unscaled model
    # for compatibility with the existing runtime pipeline.
    clf_deploy = LogisticRegression(max_iter=2000, C=1.0, random_state=42, class_weight="balanced")
    clf_deploy.fit(embeddings, labels)
    joblib.dump(clf_deploy, MODEL_PATH)
    print(f"  Model saved:  {MODEL_PATH}")
    print(f"  Features:     {clf_deploy.n_features_in_} dimensions")
    print(f"  Classes:      {clf_deploy.classes_}")

    # ── Sanity check ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SANITY CHECK")
    print("=" * 65)

    check_safe = [
        "Hey, are we still meeting for coffee tomorrow?",
        "I need to discuss the project with you.",
        "The weather is nice today, let's go for a walk.",
        "Happy birthday! Wishing you all the best.",
        "Can you send me the meeting agenda?",
        "I'll be there in 10 minutes, wait for me.",
        "Please review the document I sent earlier.",
        "Never share your OTP with anyone.",
    ]
    check_scam = [
        "Your account has been blocked. Share your OTP immediately.",
        "This is the police. Transfer money to avoid arrest.",
        "Your KYC has expired. Share your Aadhaar and PIN to update.",
        "URGENT: Your bank account will be frozen unless you verify now.",
        "You won a lottery prize of 50 lakh. Share your bank details.",
        "Install AnyDesk and share the code. I'll fix your account.",
        "Your SIM will be blocked. Share Aadhaar to verify.",
        "I'm sending you a refund. Enter your UPI PIN to receive it.",
    ]

    print("\n  Safe texts:")
    safe_ok = 0
    for t in check_safe:
        emb  = encoder.encode([t])
        prob = clf_deploy.predict_proba(emb)[0]
        pred = clf_deploy.predict(emb)[0]
        ok   = pred == 0
        safe_ok += int(ok)
        print(f"    [{'OK' if ok else 'X '}] P(scam)={prob[1]:.3f} | {t[:60]}")

    print(f"\n  Scam texts:")
    scam_ok = 0
    for t in check_scam:
        emb  = encoder.encode([t])
        prob = clf_deploy.predict_proba(emb)[0]
        pred = clf_deploy.predict(emb)[0]
        ok   = pred == 1
        scam_ok += int(ok)
        print(f"    [{'OK' if ok else 'X '}] P(scam)={prob[1]:.3f} | {t[:60]}")

    total   = len(check_safe) + len(check_scam)
    correct = safe_ok + scam_ok
    print(f"\n  Result: {correct}/{total} correct ({correct/total*100:.1f}%)")
    print("\n" + "=" * 65)
    print("  DONE! Model is ready for deployment.")
    print("=" * 65)


if __name__ == "__main__":
    main()