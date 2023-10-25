import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests  # Import the requests library to make HTTP requests
import openai

# Data cleaning function
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv('CIA_Country_Facts.csv')
    df.drop(columns='Climate', inplace=True)
    df_cleaned = df.dropna().copy()

    # Relabel countries
    country_name_mapping = {
        'Turks & Caicos Is': 'Turks & Caicos',
        'Saint Vincent and the Grenadines': 'The Grenadines',
        'St Pierre & Miquelon': 'St Pierre',
        'Korea, South': 'South Korea',
        'Saint Kitts & Nevis': 'Saint Kitts',
        'N. Mariana Islands': 'North Mariana Islands',
        'Korea, North': 'North Korea',
        'Micronesia, Fed. St.': 'Micronesia',
        'Gambia, The': 'Gambia',
        'Congo, Dem. Rep.': 'Democratic Republic of Congo',
        'Congo, Repub. of the': 'Congo',
        'Central African Rep.': 'Central African Republic',
        'British Virgin Is.': 'British Virgin Islands',
        'Bahamas, The': 'Bahamas',
        'Antigua & Barbuda': 'Antigua'
    }
    df_cleaned['Country'] = df_cleaned['Country'].replace(country_name_mapping)

    # Relabel regions
    region_name_mapping = {
        'ASIA (EX. NEAR EAST)': 'ASIA',
        'C.W. OF IND. STATES': 'INDEPENDENT STATES',
        'LATIN AMER. & CARIB': 'LATIN AMERICA',
        'NEAR EAST': 'MIDDLE EAST'
    }
    df_cleaned['Region'] = df_cleaned['Region'].str.strip().replace(region_name_mapping)

    return df_cleaned

# Generate Flag of country
def generate_image_url(country):
    image_prompt = f'Country Flag of {country}'
    response = openai.Image.create(
        prompt=image_prompt,
        n=1,
        size='256x256'
    )
    image_url = response['data'][0]['url']
    return image_url
        # For debugging
    st.write(response)
    st.write(image_url)


# Function to generate facts using GPT API
def generate_facts(country, api_key):
    headers = {
        'Authorization': f'Bearer {api_key}',
    }
    data = {
        'model': 'gpt-3.5-turbo',
        'messages': [{'role': 'system', 'content': 'You are a CIA Analyst.'},
                     {'role': 'user', 'content': f'Please give a brief synopsis of the current situation in {country}'}],
        'max_tokens': 150,
        'temperature': 0.7  # You can adjust the temperature to control the creativity of the output
    }
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
    if response.status_code == 200:
        facts = response.json()['choices'][0]['message']['content'].strip()  # Corrected path to extract the generated text
        return facts
    else:
        st.write(f'Error: {response.status_code}')
        st.write(response.text)  # This will output the error message from the API

 
def main():
    st.set_page_config(page_title="CIA Facts", layout="wide")
    
    # Load and clean data
    df_cleaned = load_and_clean_data()
    country = st.sidebar.selectbox('Select a Country:', df_cleaned['Country'].unique())
    
    # Load API key for DALL-E
    with open('key.txt', 'r') as file:
        openai.api_key = file.read().strip()
    
    # Generate and display image
    image_url = generate_image_url(country)
    
    col1, col2 = st.columns([2, 3])
    with col1:
        st.title("CIA FACTS")
    with col2:
        st.image(image_url, caption=f'{country} Image', use_column_width=True)
    

    tab1, tab2, tab3, tab4 = st.tabs(["Data", "Charts", "Current Information", "Notes"])


    with tab1:
        st.subheader('Country Fact Data')
        st.table(df_cleaned[df_cleaned['Country'] == country].T)  # Transpose the DataFrame and display it as a table

    with tab2:
        attribute_labels = {
            'Population': 'Population (in Millions)',
            'Area (sq. mi.)': 'Area (in sq. mi.)',
            'GDP ($ per capita)': 'GDP ($ per capita)',
            'Literacy (%)': 'Literacy Rate (%)',
            'Phones (per 1000)': 'Phones (per 1000 people)',
        }
        selected_attribute = st.sidebar.selectbox('Select an Attribute for Comparison:', options=list(attribute_labels.keys()))
        selected_region = df_cleaned[df_cleaned['Country'] == country]['Region'].values[0]
        region_data = df_cleaned[df_cleaned['Region'] == selected_region]

        # Visualization: Selected Attribute comparison
        plt.figure(figsize=(12, 6))
        sns.barplot(y=region_data['Country'], x=region_data[selected_attribute], palette="viridis", orient='h')
        plt.title(f'{attribute_labels[selected_attribute]} Comparison in {selected_region}')
        plt.xlabel(attribute_labels[selected_attribute])
        highlight = region_data['Country'] == country
        plt.barh(region_data[highlight].index, region_data[highlight][selected_attribute], color='red')
        plt.tight_layout()
        st.pyplot(plt)

    with tab3:
        # Load API key
        with open('key.txt', 'r') as file:
            api_key = file.read().strip()

        # More Facts Button
        if st.button('More Facts'):
            facts = generate_facts(country, api_key)
            st.write(facts)
    with tab4:  # New block for the "Notes" tab
        st.subheader('Notes')

    # Colored dialog boxes for Note 1, Note 2, Note 3 with black text
        st.markdown(
            """
            <div style="background-color: #fafafa; padding: 10px; border: 1px solid #ccc; margin-bottom: 10px; color: black;">
                <strong>Note 1:</strong> Your text for Note 1 goes here.
            </div>
            <div style="background-color: #e7f3fe; padding: 10px; border: 1px solid #ccc; margin-bottom: 10px; color: black;">
                <strong>Note 2:</strong> Your text for Note 2 goes here.
            </div>
            <div style="background-color: #ffebcc; padding: 10px; border: 1px solid #ccc; margin-bottom: 10px; color: black;">
                <strong>Note 3:</strong> Your text for Note 3 goes here.
            </div>
            """,
            unsafe_allow_html=True  # Allow HTML formatting
        )


if __name__ == "__main__":
    main()
