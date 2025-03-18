#!/usr/bin/env python3
import os
import json
import rospy                     # ROS Python client library (handles communication with ROS)
import rosservice                # Library for dynamic service type retrieval
from std_msgs.msg import String  # Standard ROS String message
from openai import OpenAI

# Initialize OpenAI client using API key from environment variables
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Global dictionary mapping face UID to name
face_names = {}

conversation_history = [
    {"role": "system", "content": (
        "You are a cultured and reliable QTrobot butler. Introduce yourself to new people, greet known individuals by their correct names, "
        "and always greet ladies first. Never forget a name, comfort those who are sad, and offer genuine compliments. "
        "Always make a compliment about someones clothes, hair, or smile. "
        "Recognize multiple people at once and detect if someone complains about being addressed incorrectly."
    )}
]


def query_gpt4(query_text):
    """
    Query GPT-4-0613 while maintaining a persistent conversation.
    (Persistent: context is kept across multiple queries.)
    """
    global conversation_history
    conversation_history.append({"role": "user", "content": query_text})
    try:
        response = client.chat.completions.create(
            model="gpt-4-0613",
            messages=conversation_history,
            max_tokens=100,
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()
        conversation_history.append({"role": "assistant", "content": answer})
        return answer
    except Exception as e:
        rospy.logerr("Error querying GPT-4: %s", e)
        return "I'm sorry, I couldn't process your request."

def call_face_detection_service():
    """
    Call the face detection service '/custom/cv/deep_face/detect' with an empty JSON.
    Returns a JSON string.
    """
    
    service_name = '/custom/cv/deep_face/detect'
    rospy.wait_for_service(service_name)
    service_class = rosservice.get_service_class_by_name(service_name)
    if service_class is None:
        rospy.logerr("Service type for %s not found.", service_name)
        return None
    try:
        proxy = rospy.ServiceProxy(service_name, service_class)
        req = service_class._request_class()
        resp = proxy(req)
        return resp.json_result  # Returns a JSON string
    except Exception as e:
        rospy.logerr("Face detection service call failed: %s", e)
        return None

def parse_face_detection(json_str):
    """
    Parse the JSON string from the face detection service into a dictionary.
    (Dictionary: a data structure mapping keys to values.)
    """
    try:
        data = json.loads(json_str)
        filtered_data = {k: v for k, v in data.items() if v is not None}
        return filtered_data
    except Exception as e:
        rospy.logerr("Error parsing face detection data: %s", e)
        return {}

def call_microphone_service():
    """
    Call the microphone recognition service to capture spoken input.
    """
    print("Listening...")
    service_name = '/custom/speech/sr/microphone_recognize'
    rospy.wait_for_service(service_name)
    service_class = rosservice.get_service_class_by_name(service_name)
    if service_class is None:
        rospy.logerr("Service type for %s not found.", service_name)
        return ""
    try:
        proxy = rospy.ServiceProxy(service_name, service_class)
        req = service_class._request_class()
        req.language = "en-US"
        resp = proxy(req)
        return resp.text.strip()  # Assumes recognized text is in 'text'
    except Exception as e:
        rospy.logerr("Microphone service call failed: %s", e)
        return ""

def call_tts_service(tts_text):
    """
    Call the TTS service '/qt_robot/speech/say' to speak the provided text.
    (TTS: text-to-speech conversion.)
    """
    service_name = '/qt_robot/speech/say'
    rospy.wait_for_service(service_name)
    service_class = rosservice.get_service_class_by_name(service_name)
    if service_class is None:
        rospy.logerr("TTS service type for %s not found.", service_name)
        return False
    try:
        proxy = rospy.ServiceProxy(service_name, service_class)
        req = service_class._request_class()
        req.message = tts_text
        resp = proxy(req)
        return True
    except Exception as e:
        rospy.logerr("TTS service call failed: %s", e)
        return False

def update_face_names(mic_text, face_data):
    """
    Update the face_names dictionary if the microphone input contains "my name is <name>".
    Then persist the updated dictionary.
    """
    global face_names
    lower_text = mic_text.lower()
    if "my name is" in lower_text:
        try:
            name_part = lower_text.split("my name is", 1)[1].strip()
            name = name_part.split()[0].capitalize()
        except Exception as e:
            rospy.logerr("Error parsing name from microphone text: %s", e)
            return
        for uid in face_data.keys():
            if face_names.get(uid, "unknown") == "unknown":
                face_names[uid] = name
                rospy.loginfo("Updated face UID %s with name %s", uid, name)
                # Save updated dictionary immediately
                save_face_names()
                break

def construct_query(mic_text, face_data):
    """
    Construct the GPT-4 query by including butler instructions,
    the spoken input, and a summary of the face detection data.
    """
    instructions = (
        "You are a cultured and reliable QTrobot butler. "
        "Introduce yourself to new people, greet known individuals by their correct names, and always greet ladies first. "
        "Never forget a name, comfort those who are sad, and offer genuine compliments. "
        "Recognize multiple people at once and detect if someone complains about being addressed incorrectly."
    )
    if mic_text:
        instructions += f"\nSpoken input: '{mic_text}'."
    if face_data:
        faces_summary = ""
        num_faces = len(face_data.keys())
        faces_summary += f"\nNumber of detected persons: {num_faces}."
        for uid, faces in face_data.items():
            name = face_names.get(uid, "unknown")
            for face in faces:
                gender = face.get("dominant_gender", "unknown")
                emotion = face.get("dominant_emotion", "unknown")
                faces_summary += f" (Face {uid}: {name}, {gender}, {emotion})."
                break
        instructions += "\nFace detection data:" + faces_summary
    instructions += "\nProvide a short, concise response as the butler."
    return instructions

def greet_faces(face_data):
    """
    Return a greeting based on detected faces and known names.
    """
    if not face_data:
        return "Hi, whats your name?"
    names = ''
    for uid in face_data.keys():
        name = face_names.get(uid, "unknown")
        if name != "unknown":
            if names:
                names += " and "
            names += name
    if not names:
        return "Hi, whats your name?"
    return f"Hello {names}, welcome back!"

if __name__ == '__main__':
    rospy.init_node("qtrobot_butler_node", anonymous=True)
    rospy.loginfo("QTrobot Butler Node started.")
    rate = rospy.Rate(0.05)  # every 20 seconds

    def shutdown_hook():
        rospy.loginfo("Shutting down QTrobot Butler Node gracefully.")
        save_face_names()

    rospy.on_shutdown(shutdown_hook)

    try:
        while not rospy.is_shutdown():
            rospy.loginfo("Starting face detection...")
            face_json = call_face_detection_service()
            if face_json:
                face_data = parse_face_detection(face_json)
                rospy.loginfo("Face detection data: %s", face_data)
            else:
                rospy.loginfo("No face detection data received.")
                face_data = {}

            if not face_data:
                rospy.loginfo("No faces detected. Skipping greeting and listening.")
                rate.sleep()
                continue

            for uid in face_data.keys():
                if uid not in face_names:
                    face_names[uid] = "unknown"

            greeting = greet_faces(face_data)
            rospy.loginfo("Greeting: %s", greeting)
            call_tts_service(greeting)

            rospy.sleep(3)

            rospy.loginfo("Starting microphone capture...")
            mic_text = call_microphone_service()
            rospy.loginfo("Captured spoken input: %s", mic_text)

            update_face_names(mic_text, face_data)

            query_text = construct_query(mic_text, face_data)
            rospy.loginfo("GPT-4 query: %s", query_text)

            answer = query_gpt4(query_text)
            rospy.loginfo("GPT-4 answer: %s", answer)

            call_tts_service(answer)

            rate.sleep()

    except KeyboardInterrupt:
        rospy.loginfo("KeyboardInterrupt detected. Exiting gracefully.")