import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { DiaryComponent } from './diary/diary.component';
import { ChatbotComponent } from './chatbot/chatbot.component';

const routes: Routes = [
  {path: 'diary', component: DiaryComponent},
  {path: 'chatbot', component: ChatbotComponent}
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
