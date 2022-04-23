mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"lntri@csc.hcmus.edu.vn\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml