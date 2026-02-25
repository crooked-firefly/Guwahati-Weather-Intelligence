import streamlit as st

def check_password():
    """Returns `True` if the user entered the correct credentials."""

    def password_entered():
        """Validates the entered username and password."""
        entered_user = st.session_state.get("username", "")
        entered_password = st.session_state.get("password", "")

        if entered_user == "admin" and entered_password == "admin2001":
            st.session_state["password_correct"] = True
            st.session_state["current_user"] = entered_user
        else:
            st.session_state["password_correct"] = False

    # --- Login UI ---
    if "password_correct" not in st.session_state or not st.session_state["password_correct"]:
        st.markdown("""
            <div style="text-align: center; padding: 2rem 1rem;">
                <h1 style="color: #1f77b4;">🌦️ Guwahati Daily Weather Dashboards</h1>
                <p style="font-size: 1.1rem;">Secure access to climate intelligence tools</p>
                <img src="https://cdn-icons-png.flaticon.com/512/1779/1779940.png" width="100" style="margin-top: 1rem;"/>
            </div>
        """, unsafe_allow_html=True)

        st.text_input("👤 Username", key="username")
        st.text_input("🔒 Password", type="password", on_change=password_entered, key="password")

        if "password_correct" in st.session_state and not st.session_state["password_correct"]:
            st.error("😕 User not known or password incorrect")

        st.markdown("""
            <hr style="margin-top: 3rem;"/>
            <div style="text-align: center; font-size: 0.9rem; color: gray;">
                Created by <strong>Aman Kumar</strong> | 📧 <a href='mailto:amancrazy2001@gmail.com'>amancrazy2001@gmail.com</a>
            </div>
        """, unsafe_allow_html=True)

        return False

    else:
        return True
