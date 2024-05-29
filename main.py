import streamlit as st
from streamlit_option_menu import option_menu
import streamlit as st
import pages.quiz_creation_page
import pages.quiz_solve_page
import pages.quiz_grading_page
import sign
from streamlit_option_menu import option_menu
import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Pondering")

st.markdown(
    """
    <!-- Global site tag (gtag.js) - Google Analytics -->
    <script async src=f"https://www.googletagmanager.com/gtag/js?id={os.getenv('analytics_tag')}"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', os.getenv('analytics_tag'));
    </script>
    """, unsafe_allow_html=True)

class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        with st.sidebar:
            app = option_menu(
                menu_title='Pondering',
                options=['Home', 'Account', 'Trending', 'Your Posts', 'About', 'Buy Me a Coffee'],
                icons=['house-fill', 'person-circle', 'trophy-fill', 'chat-fill', 'info-circle-fill', 'cup-fill'],
                menu_icon='chat-text-fill',
                default_index=0,
                styles={
                    "container": {"padding": "5!important", "background-color": 'black'},
                    "icon": {"color": "white", "font-size": "23px"},
                    "nav-link": {"color": "white", "font-size": "20px", "text-align": "left", "margin": "0px", "--hover-color": "blue"},
                    "nav-link-selected": {"background-color": "#02ab21"},
                }
            )

        if app == "Home":
            home.app()
        elif app == "Account":
            account.app()
        elif app == "Trending":
            trending.app()
        elif app == 'Your Posts':
            your.app()
        elif app == 'About':
            about.app()
        elif app == 'Buy Me a Coffee':
            buy_me_a_coffee.app()

multi_app = MultiApp()
multi_app.add_app("Home", home.app)
multi_app.add_app("Account", account.app)
multi_app.add_app("Trending", trending.app)
multi_app.add_app("Your Posts", your.app)
multi_app.add_app("About", about.app)
multi_app.add_app("Buy Me a Coffee", buy_me_a_coffee.app)
multi_app.run()
