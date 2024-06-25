import os
import streamlit as st
import requests
import os
from autogen import ConversableAgent
from openai import OpenAI
from autogen import register_function
from src.func import ToolFunctions, Autogen_tools #generate_image_with_dalle, get_weather

autogen_tools = Autogen_tools()
def main():
    st.title("Weather based Food Recipe")

    # User input for town
    town = st.text_input("Enter a city name:")

    if st.button("Generate"):
        if town:
            try:
                # Initiate chat with Autogen
                chat_result = None
                user_proxy, assistant = autogen_tools.create_autogen_assistant()
                chat_result = user_proxy.initiate_chat(assistant, message=town)
                # print(chat_result)
                # Process chat result to extract recipe and image URL
                recipe_content = autogen_tools.process_chat_result(chat_result)
                
                # Display recipe
                if recipe_content:
                    st.subheader("Recipe:")
                    st.markdown(recipe_content)
                else:
                    st.subheader("There is an issue:")
                    st.markdown("The town is unkown, please try again !")
                
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()