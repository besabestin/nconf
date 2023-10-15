import { Component, OnInit } from '@angular/core';
import { StoryService } from './story.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'personal-guide';
  constructor(private storyService : StoryService) { }
  ngOnInit(): void { 
    this.storyService.getStories()
    // just a quick soln. beause openai takes a while to load
    // we load some part of it after few seconds
    setTimeout(() => {
      this.storyService.getTasks()
    }, 2000);
  }
}
