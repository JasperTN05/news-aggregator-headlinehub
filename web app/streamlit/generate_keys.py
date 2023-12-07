import streamlit_authenticator as stauth

hashed_passwords = stauth.Hasher(['xxx', 'xxx']).generate()
for i in hashed_passwords:
    print(i)


