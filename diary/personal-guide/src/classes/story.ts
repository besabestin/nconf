export class Life {
    stories?: Story[];
    allTasks?: AllTasks;
}

export class Story {
    sentiment?: string;
    events?: string;
    when?: string;
    content?: string;
    images?: string[];
}

export class SearchResult {
    message?: string;
}

export class Visualizer {
    images?: string[];
}

export class DateTasks {
    date?: string;
    events?: string[];
}

export class AllTasks {
    tasks?: DateTasks[];
    today?: DateTasks;
}