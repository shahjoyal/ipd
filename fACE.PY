import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import face_recognition
import time

 
webcam_video_stream = cv2.VideoCapture(0)


face_exp_model = model_from_json(open("facial_expression_model_structure.json","r",encoding="utf-8").read())
face_exp_model.load_weights('facial_expression_model_weights.h5')


emotions_label = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')


neutral_count = 0
happy_count = 0
sad_count = 0
start_time = time.time()


while True:
    
    ret, current_frame = webcam_video_stream.read()
    
    
    current_frame_small = cv2.resize(current_frame, (0,0), fx=0.25, fy=0.25)
    
    
    all_face_locations = face_recognition.face_locations(current_frame_small, number_of_times_to_upsample=2, model='hog')
    

    for current_face_location in all_face_locations:
        
        top_pos, right_pos, bottom_pos, left_pos = [i*4 for i in current_face_location]
        current_face_image = current_frame[top_pos:bottom_pos, left_pos:right_pos]

        
        current_face_image_gray = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY) 
        current_face_image_gray = cv2.resize(current_face_image_gray, (48, 48))
        img_pixels = image.img_to_array(current_face_image_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255 

        
        exp_predictions = face_exp_model.predict(img_pixels) 
        max_index = np.argmax(exp_predictions[0])
        emotion_label = emotions_label[max_index]
        
    
        if emotion_label == 'neutral':
            neutral_count += 1
        elif emotion_label == 'happy':
            happy_count += 1
        elif emotion_label == 'sad':
            sad_count += 1
        
        
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0, 255, 0), 2)
        cv2.putText(current_frame, emotion_label, (left_pos, top_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    
    elapsed_time = time.time() - start_time
    if elapsed_time >= 10:  
        break

    
    cv2.imshow("Webcam Video", current_frame)

    
    cv2.waitKey(1)


print("Neutral count:", neutral_count)
print("Happy count:", happy_count)
print("Sad count:", sad_count)


webcam_video_stream.release()
cv2.destroyAllWindows()
if neutral_count>20:
    print("attentive")
    import spacy
    nlp = spacy.load('en_core_web_md')
    word1 = nlp.vocab['wolf'].vector
    word2 = nlp.vocab['dog'].vector
    word3 = nlp.vocab['cat'].vector
    from scipy import spatial
    cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x, y)
    new_vector = word1 - word2 + word3
    computed_similarities = []
    for word in nlp.vocab:
        if word.has_vector:
            if word.is_lower:
                if word.is_alpha:
                    similarity = cosine_similarity(new_vector, word.vector)
                    computed_similarities.append((word, similarity))
    computed_similarities = sorted(computed_similarities, key=lambda item: -item[1])
    def vector_math(a,b,c):
        new_vector = nlp.vocab[a].vector - nlp.vocab[b].vector + nlp.vocab[c].vector
        computed_similarities = []
        for word in nlp.vocab:
            if word.has_vector:
                if word.is_lower:
                    if word.is_alpha:
                        similarity = cosine_similarity(new_vector, word.vector)
                        computed_similarities.append((word, similarity))

        computed_similarities = sorted(computed_similarities, key=lambda item: -item[1])

        return [w[0].text for w in computed_similarities[:10]]
    import nltk
    nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    review = input("Enter the review about the lecture ")
    sid.polarity_scores(review)
    def review_rating(string):
        scores = sid.polarity_scores(string)
        if scores['compound'] == 0:
            return 'Neutral'
        elif scores['compound'] > 0:
            return 'Positive'
        else:
            return 'Negative'
    print("the review is classified as")
    print(review_rating(review))

else:
    print("not attentive you cannot give your review!")

