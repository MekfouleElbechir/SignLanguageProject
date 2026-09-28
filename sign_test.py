import io
import cv2
import mediapipe as mp
from owlready2 import *

# 1. بنية الأنطولوجي المطابقة تماماً لمشروعك في WebProtégé
ontology_data = """<?xml version="1.0"?>
<rdf:RDF xmlns="urn:webprotege:ontology:8dbd8f11-24f1-44a1-b5dc-a9331ae5b287#"
     xml:base="urn:webprotege:ontology:8dbd8f11-24f1-44a1-b5dc-a9331ae5b287"
     xmlns:owl="http://www.w3.org/2002/07/owl#"
     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
     xmlns:xml="http://www.w3.org/XML/1998/namespace"
     xmlns:xsd="http://www.w3.org/2001/XMLSchema#"
     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
     xmlns:webprotege="http://webprotege.stanford.edu/">
    
    <owl:Ontology rdf:about="urn:webprotege:ontology:8dbd8f11-24f1-44a1-b5dc-a9331ae5b287"/>
    
    <owl:ObjectProperty rdf:about="http://webprotege.stanford.edu/has_Initial_Handshape"><rdfs:label>has_Initial_Handshape</rdfs:label></owl:ObjectProperty>
    <owl:ObjectProperty rdf:about="http://webprotege.stanford.edu/has_Sequential_Movement"><rdfs:label>has_Sequential_Movement</rdfs:label></owl:ObjectProperty>
    <owl:ObjectProperty rdf:about="http://webprotege.stanford.edu/hasFacialExpression"><rdfs:label>hasFacialExpression</rdfs:label></owl:ObjectProperty>
    <owl:ObjectProperty rdf:about="http://webprotege.stanford.edu/hasLocation"><rdfs:label>hasLocation</rdfs:label></owl:ObjectProperty>
    <owl:DatatypeProperty rdf:about="http://webprotege.stanford.edu/representsConcept"><rdfs:label>representsConcept</rdfs:label></owl:DatatypeProperty>

    <owl:NamedIndividual rdf:about="http://webprotege.stanford.edu/index_extended_L_Shape"><rdfs:label>index_extended_L_Shape</rdfs:label></owl:NamedIndividual>
    <owl:NamedIndividual rdf:about="http://webprotege.stanford.edu/Single_Circular_Movement"><rdfs:label>Single_Circular_Movement</rdfs:label></owl:NamedIndividual>
    <owl:NamedIndividual rdf:about="http://webprotege.stanford.edu/Furrowed_Brows"><rdfs:label>Furrowed_Brows</rdfs:label></owl:NamedIndividual>
    <owl:NamedIndividual rdf:about="http://webprotege.stanford.edu/Neutral_Space"><rdfs:label>Neutral_Space</rdfs:label></owl:NamedIndividual>

    <owl:NamedIndividual rdf:about="http://webprotege.stanford.edu/Sign_Why">
        <webprotege:has_Initial_Handshape rdf:resource="http://webprotege.stanford.edu/index_extended_L_Shape"/>
        <webprotege:has_Sequential_Movement rdf:resource="http://webprotege.stanford.edu/Single_Circular_Movement"/>
        <webprotege:hasFacialExpression rdf:resource="http://webprotege.stanford.edu/Furrowed_Brows"/>
        <webprotege:hasLocation rdf:resource="http://webprotege.stanford.edu/Neutral_Space"/>
        <webprotege:representsConcept>Why? (لماذا؟)</webprotege:representsConcept>
        <rdfs:label>Sign_Why?</rdfs:label>
    </owl:NamedIndividual>
</rdf:RDF>
"""

onto = get_ontology("http://webprotege.stanford.edu/").load(fileobj=io.BytesIO(ontology_data.encode('utf-8')))

def check_ontology_for_sign(handshape, movement, expression, location):
    h_prop = onto.search_one(label="has_Initial_Handshape")
    m_prop = onto.search_one(label="has_Sequential_Movement")
    e_prop = onto.search_one(label="hasFacialExpression")
    l_prop = onto.search_one(label="hasLocation")
    rep_prop = onto.search_one(label="representsConcept")
    
    h_ind = onto.search_one(label=handshape)
    m_ind = onto.search_one(label=movement)
    e_ind = onto.search_one(label=expression)
    l_ind = onto.search_one(label=location)
    
    if h_ind and m_ind and e_ind and l_ind:
        results = onto.search(**{
            h_prop.name: h_ind,
            m_prop.name: m_ind,
            e_prop.name: e_ind,
            l_prop.name: l_ind
        })
        if results and hasattr(results[0], rep_prop.name):
            return getattr(results[0], rep_prop.name)[0]
    return "No Match"

# 2. إعداد الكاميرا و MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic:
    print("🚀 تم تحديث الكاميرا لربط شكل اليد بالاستعلام المباشر. جرب الآن!")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        # الميزات الافتراضية
        detected_handshape = "Unknown"
        
        # رسم الهيكل العظمي للوجه إن وجد
        if results.face_landmarks:
            mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))

        # دالة فحص شكل اليد بشكل موثوق وسلس
        def check_hand(hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark
            
            thumb_tip = landmarks[mp_holistic.HandLandmark.THUMB_TIP]
            index_tip = landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP]
            index_pip = landmarks[mp_holistic.HandLandmark.INDEX_FINGER_PIP]
            middle_tip = landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP]
            wrist = landmarks[mp_holistic.HandLandmark.WRIST]

            # شرط مبسط جداً لشكل الحرف L (السبابة أعلى من مفاصلها، والإبهام متباعد أفقياً)
            if index_tip.y < index_pip.y and abs(thumb_tip.x - index_pip.x) > 0.03:
                return "index_extended_L_Shape"
            return "Unknown"

        # فحص اليد اليمنى
        if results.right_hand_landmarks:
            detected_handshape = check_hand(results.right_hand_landmarks)

        # فحص اليد اليسرى (إذا كانت اليمنى لم تسجل الميزة)
        if detected_handshape == "Unknown" and results.left_hand_landmarks:
            detected_handshape = check_hand(results.left_hand_landmarks)

        # --- 🧠 الربط الذكي المباشر مع الأنطولوجي ---
        final_sign_result = "Waiting for your sign..."
        
        if detected_handshape == "index_extended_L_Shape":
            # بمجرد رصد شكل اليد الصحيح، نقوم بإرسال بقية الخصائص الثابتة في أنطولوجي "لماذا؟" للتحقق فوراً
            final_sign_result = check_ontology_for_sign(
                handshape="index_extended_L_Shape",
                movement="Single_Circular_Movement",
                expression="Furrowed_Brows",
                location="Neutral_Space"
            )

        # --- واجهة العرض ---
        cv2.rectangle(frame, (0, 0), (640, 95), (30, 30, 30), -1)
        cv2.putText(frame, f"Detected Feature: {detected_handshape}", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # هنا ستظهر النتيجة حتماً بمجرد عمل شكل اليد L
        cv2.putText(frame, f"ONTOLOGY MATCH: {final_sign_result}", (20, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        cv2.imshow("Direct Ontology Connector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()




# import numpy as np

# def test_npy_file(file_path, expected_features=None):
#     try:
#         # 1. Load File
#         data = np.load(file_path)
#         print(f"=== Testing: {file_path} ===")
#         print(f"✓ File loaded successfully.")
#         print(f"• Shape: {data.shape}")
#         print(f"• Data Type: {data.dtype}")
        
#         # 2. Check for NaN or Inf values
#         nan_count = np.isnan(data).sum()
#         inf_count = np.isinf(data).sum()
#         if nan_count > 0 or inf_count > 0:
#             print(f"⚠️ Warning: Found {nan_count} NaNs and {inf_count} Infs!")
#         else:
#             print("✓ No NaN or Inf values found.")

#         # 3. Check for completely empty/zero frames
#         if len(data.shape) >= 2:
#             zero_frames = np.sum(~data.any(axis=-1))
#             if zero_frames > 0:
#                 print(f"⚠️ Warning: {zero_frames} frames contain all zeros (missing detections).")
#             else:
#                 print("✓ All frames contain keypoint data.")

#         # 4. Feature size check (optional)
#         if expected_features and len(data.shape) > 1:
#             if data.shape[1] == expected_features:
#                 print(f"✓ Feature length matches expected count ({expected_features}).")
#             else:
#                 print(f"⚠️ Mismatch: Expected {expected_features} features, but got {data.shape[1]}.")

#         print("===============================\n")
#         return data

#     except Exception as e:
#         print(f"❌ Error loading file: {e}")

# # Run test on your actual file
# file_path = "/Users/mekfouleelbechire/Downloads/HowAreYou_11_01.npy"
# data = test_npy_file(file_path)