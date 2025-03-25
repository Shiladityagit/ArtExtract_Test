from qdrant_client import QdrantClient
from io import BytesIO
import streamlit as st
import base64

collection_name = "paintings_images"

if 'selected_record' not in st.session_state:
    st.session_state.selected_record = None

def set_selected_record(new_record):
    st.session_state.selected_record = new_record

@st.cache_resource
def get_client():
    return QdrantClient(
        url=st.secrets["QDRANT_URL"], 
        api_key=st.secrets["QDRANT_API_KEY"]
    )


def get_initial_records():
    client = get_client()

    records, _ = client.scroll(
        collection_name=collection_name,
        with_vectors = False,
        limit = 12
    )
    return records

def get_similar_search():
    client = get_client()
    if st.session_state.selected_record is not None:
        return client.recommend(
            collection_name=collection_name,
            positive=[st.session_state.selected_record.id],
            limit=12
        )
    return []

def get_bytes_from_base64(base64_string):
    return BytesIO(base64.b64decode(base64_string))

records = get_similar_search() if st.session_state.selected_record is not None else get_initial_records()

if st.session_state.selected_redord:
    image_bytes = get_bytes_from_base64(
        st.session_state.selected_record.payload["base64"])
    st.header("Images Similar to:")
    st.image(
        image=image_bytes
    )
    st.divider()

column = st.columns(3)

for idx, record in enumerate(records):
    col_idx = idx % 3
    image_bytes = get_bytes_from_base64(record.payload["base64"])
    with column[col_idx]:
        st.image(
            image = image_bytes
        )
        st.button(
            label = "Find similar images",
            key = record.id,
            on_click = set_selected_record,
            args = [record]
        )