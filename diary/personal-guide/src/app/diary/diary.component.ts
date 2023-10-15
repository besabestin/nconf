import { Component, OnInit } from '@angular/core';
import { Story } from 'src/classes/story';
import { StoryService } from '../story.service';

@Component({
  selector: 'app-diary',
  templateUrl: './diary.component.html',
  styleUrls: ['./diary.component.css']
})
export class DiaryComponent implements OnInit {

  storyDate = "";
  storyContent = "";
  searchEntry = "";
  showJournalCreator = false;

  constructor(public storyService : StoryService) { }

  ngOnInit(): void { }

  saveStory(){
    this.storyService.addStory(this.storyDate, this.storyContent);
    this.storyDate = "";
    this.storyContent = "";
  }

  searchInStory(){
    this.storyService.searchInJournal(this.searchEntry);
  }

  searchImages(aStory: Story){
    this.storyService.getImages(aStory)
  }

}
