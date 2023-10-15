import { Component, OnInit } from '@angular/core';
import { StoryService } from '../story.service';

@Component({
  selector: 'app-chatbot',
  templateUrl: './chatbot.component.html',
  styleUrls: ['./chatbot.component.css']
})
export class ChatbotComponent implements OnInit {
  newMessage = "";
  constructor(public storyService : StoryService) { }

  ngOnInit(): void {
  }

  sendMessage() {
    if (this.newMessage.trim() !== ''){
      this.storyService.chatMessages.push(this.newMessage)
      this.storyService.getChatResponse()
    }
  }

}
