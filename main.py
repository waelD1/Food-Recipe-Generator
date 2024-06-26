import os
import streamlit as st
import requests
import os
from autogen import ConversableAgent
from openai import OpenAI
from autogen import register_function
from src.autogen_functions import ToolFunctions, Autogen_tools #generate_image_with_dalle, get_weather

# tool_functions = ToolFunctions()
autogen_tools = Autogen_tools()
def main():
    st.title("Weather based Food Recipe")

    # User input for town
    town = st.text_input("Enter a city name:")

    if st.button("Generate"):
        if town:
            try:
                ## Initiate chat with Autogen

                chat_result = None
                # Initiate the assistant
                user_proxy, assistant = autogen_tools.create_autogen_assistant()
                # Initiate the chat with the town's name
                chat_result = user_proxy.initiate_chat(assistant, message=town)
                # Use the chat_history to extract the recipe, image and the recommendations
                recipe_content, image_url, yelp_reco = autogen_tools.process_chat_result(chat_result)
                
                # Display recipe
                if recipe_content:
                    st.subheader("Recipe:")
                    st.markdown(recipe_content)
                else:
                    st.subheader("There is an issue:")
                    st.markdown("The town is unkown, please try again !")
                
                # Display image
                if image_url:
                    st.subheader("Generated Image:")
                    st.image(image_url, caption="Generated Image", use_column_width=True)

                # Display Yelp recommendations
                if yelp_reco:
                    st.subheader("Restaurent recommended by Yelp :")
                    st.markdown(f"{yelp_reco}")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()