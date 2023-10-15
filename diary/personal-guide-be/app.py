import os
import json
import openai
from flask import Flask, request, jsonify
import logging
import datetime

logging.basicConfig(level=logging.INFO)

openai.api_key = os.getenv("OPENAI_API_KEY")
stories_fn = "ministories.json"

app = Flask(__name__)


@app.route('/')
def hello_world():
    print('running completion')
    print(openai.api_key)
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Who was the first president of the US?"}
        ]
    )
    print(completion)
    #return f'Hello World!{completion["choices"][0]["message"]["content"]}'
    response = jsonify(completion)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


def corsify_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response


@app.route('/chat', methods=["POST", "OPTIONS"])
def respond_to_chat():
    """
    api endpoint to respond to chatbot requests
    returns completion coming from openai api
    """
    
    if request.method == 'POST':
        reqbody = request.json
        ret = ""
        if 'messages' in reqbody and len(reqbody['messages']) > 0:
            lastprompt = reqbody['messages'][-1]
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": f"reply to the following prompt as if you are an annoying chatbot, reply annoyed: {lastprompt}"}
                ]
            )
            ret = completion["choices"][0]["message"]["content"]
        return corsify_response(jsonify({"message": ret}))
    
    return corsify_response(jsonify({"message": "options"}))


def get_sentiment(content):
    #connects to openai api to get sentiment
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user", 
                "content": f"""
                    Get the sentiment out of this diary content using a maximum of 5 emoticons and using no text: {content}"""
            }
        ]
    )
    ret = completion["choices"][0]["message"]["content"]
    return ret


def extract_tasks(content):
    fn_spec = {
        "name": "add_activity_date",
        "description": "you are given a long journal text that starts with today's date. The function should extract calendar entries that would happen in the future from the text. Things that already happened should be ignored. if the journal date is 1st Sep 2021 and it contains 'do something in 2 days' then the task is 'do something' and  the date is 03/09/2021. if the text contains 'grandma has birthday on the 5th of Nov' then the task is 'grandmas birthday' and the date is the 05/11/2021.",
        "parameters": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "The task that needs to be done in the coming future"
                            },
                            "date": {
                                "type": "string",
                                "description": "The exact calendar date of the task that will happen in the future. date should be in an iso format and should contain numbers and dash like YYYY-MM-DD."
                            },
                        }
                    }
                }
            }
        }
    }

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages = [
            {"role": "user", "content": content}
        ],
        functions = [fn_spec]
    )
    try:
        args = json.loads(completion["choices"][0]["message"]["function_call"]["arguments"])
        logging.info(args)
        return args
    except KeyError:
        return {'tasks': []}

def proper_date(date_str):
    try:
        datetime.date.fromisoformat(date_str)
        return True
    except ValueError:
        return False

def organize_calendar_entries(contents):
    all_events = {}
    evt_groups = [extract_tasks(f'today is {story["when"]}. {story["content"]}') for story in contents]
    for evt_group in evt_groups:
        if "tasks" in evt_group:
            evts = evt_group["tasks"] #returns an array of "task" and "date"
            for evt in evts:
                if proper_date(evt['date']):
                    if evt['date'] not in all_events:
                        all_events[evt['date']] = []
                    all_events[evt['date']].append(evt['task'])
                else:
                    logging.info(f'invalid date {evt["date"]}')
    logging.info(all_events)
    sorted_dates = sorted(all_events.keys())
    sorted_events = []
    today_events = {'date': datetime.date.today().isoformat(), 'events': []}
    for _date in sorted_dates:
        sorted_events.append({
            'date': _date,
            'events': all_events[_date]
        })
        if _date == datetime.date.today().isoformat():
            today_events = {
                'date': _date,
                'events': all_events[_date]
            }

    return sorted_events, today_events

def generate_story_stat(story):
    sentiment = get_sentiment(story["content"])
    #sentiment = "sentiment placeholder"
    return {
        "sentiment": sentiment,
        "when": story["when"],
        "content": story["content"]
    }


def generate_story_stats(stories):
    return [generate_story_stat(x) for x in stories]


@app.route('/newstory', methods=["POST", "OPTIONS"])
def add_story():
    if request.method == 'POST':
        print(request.json)
        reqbody = request.json
        story = {
            "when": reqbody["when"],
            "content": reqbody["content"]
        }
        story = generate_story_stat(reqbody)
        
        return corsify_response(jsonify(story))
    
    return corsify_response(jsonify({"message": "options"}))


@app.route('/tasks')
def get_tasks():
    stories = []
    with open(stories_fn, 'r') as f:
        stories = json.load(f)
    sorted_events, today_events = organize_calendar_entries(stories)
    return corsify_response(jsonify({"tasks": sorted_events, "today": today_events}))


@app.route('/stories')
def get_stories():
    stories = []
    with open(stories_fn, 'r') as f:
        stories = json.load(f)
    return corsify_response(jsonify({"stories": generate_story_stats(stories)}))


@app.route('/searchjournal', methods=["POST", "OPTIONS"])
def search_in_journal():
    if request.method == 'POST':
        reqbody = request.json
        print(reqbody)
        content = ""
        if reqbody['diary'] and len(reqbody['diary']) > 0:
            for diaryentry in reqbody['diary']:
                content += f"{diaryentry['when']} {diaryentry['content']}"
            prompt = reqbody['entry']
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user", 
                        "content": f"""
                        I ll give you diary entries for several days and 
                        you will answer question based on the diary entries. 
                        I ll mark the start of the question with the word Qn. 
                        The diary entries will be date entries followed by text. 
                        {content} Qn: {prompt}"""
                    }
                ]
            )
            ret = completion["choices"][0]["message"]["content"]
            return corsify_response(jsonify({"message": ret}))
    return corsify_response(jsonify({"message": "options"}))


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5153)

# in mac
# echo 'export OPENAI_API_KEY=<>' >> ~/.zshrc'
# source ~/.zshrc
