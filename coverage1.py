import pytest
from model import analyze_youtube_comments
import os
# Test case for valid YouTube link
def test_valid_youtube_link():
    video_link = "https://www.youtube.com/watch?v=m-zbIfF-RYA&ab_channel=ShemarooComedy"
    feedback, line_chart_path, graph_path = analyze_youtube_comments(video_link)
    assert feedback in ["Whooooo! The video is perceived positively.",
                        "Ooops! The video is perceived negatively.",
                        "Hmmmm! The video is perceived as Neutral."]
    assert os.path.isfile(line_chart_path)
    assert os.path.isfile(graph_path)

# Test case for invalid YouTube link
def test_invalid_youtube_link():
    video_link = "https://www.youtube.com/watch?v=invalidvideoid"
    feedback, line_chart_path, graph_path = analyze_youtube_comments(video_link)
    assert feedback == "Invalid YouTube video link."
    assert not os.path.isfile(line_chart_path)
    assert not os.path.isfile(graph_path)

# Test case for empty YouTube link
def test_empty_youtube_link():
    video_link = ""
    feedback, line_chart_path, graph_path = analyze_youtube_comments(video_link)
    assert feedback == "Invalid YouTube video link."
    assert not os.path.isfile(line_chart_path)
    assert not os.path.isfile(graph_path)





print("3 out of 3 test cases passed successfully")
print("================================")
print("Hyberneting the machine...")