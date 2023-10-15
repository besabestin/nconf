import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { AllTasks, Life, Visualizer } from 'src/classes/story';
import { ChatResponse } from 'src/classes/chatresponse';
import { Observable, map, of } from 'rxjs';
import { Story } from 'src/classes/story';
import { SearchResult } from 'src/classes/story';

@Injectable({
  providedIn: 'root'
})
export class StoryService {
  url = "http://localhost:5153/stories";
  chatUrl = "http://localhost:5153/chat";
  newStoryUrl = "http://localhost:5153/newstory";
  searchUrl = "http://localhost:5153/searchjournal";
  visualizeUrl = "http://localhost:5153/visualizer";
  tasksUrl = "http://localhost:5153/tasks"

  aLife?: Life;

  chatMessages : string[] = [
    "Hello! I am your own personal assistant. Ask away anything"
  ];

  searchResult = ""

  constructor(private http: HttpClient) { }
  
  getStories(){
    this.http.get<Life>(this.url)
    .subscribe(theLife => {
      if(theLife){
        this.aLife = theLife;
      }
    });
  }

  getTasks(){
    this.http.get<AllTasks>(this.tasksUrl)
    .subscribe(tasks => {
      if(tasks && this.aLife){
        this.aLife.allTasks = tasks;
        console.log('loading tasks done')
        console.log(this.aLife)
      }
    });
  }

  addStory(storyDate: string, storyContent: string){
    //todo normally add the one coming from backend
    let story = {when: storyDate, content: storyContent}
    this.http.post<Story>(this.newStoryUrl, story, {'headers': { 'Content-Type': 'application/json'} })
    .subscribe(aStory => {
      if (aStory){
        console.log(aStory)
        this.aLife?.stories?.unshift(aStory);
      }
    });
  }

  getChatResponse(){
    this.http.post<ChatResponse>(this.chatUrl, {"messages": this.chatMessages}, {'headers': { 'Content-Type': 'application/json'} })
    .subscribe(aReply => {
      if(aReply){
        if (aReply.message !== '')
          this.chatMessages.push(aReply.message!)
      }
    });
  }

  searchInJournal(searchMessage: string){
    this.http.post<SearchResult>(this.searchUrl, {"entry": searchMessage, "diary": this.aLife?.stories}, {'headers': { 'Content-Type': 'application/json'} })
    .subscribe(aResult => {
      if(aResult && aResult.message !== ''){
        this.searchResult = aResult.message!;
      }
    });
  }

  getImages(aStory: Story){
    this.http.post<Visualizer>(this.visualizeUrl, {"content": aStory.events}, {'headers': { 'Content-Type': 'application/json'} })
    .subscribe(imageUrls => {
      console.log(imageUrls);
      if(imageUrls && imageUrls.images && imageUrls.images.length != 0){
        aStory.images = imageUrls.images;
      }
    });
  }
}
