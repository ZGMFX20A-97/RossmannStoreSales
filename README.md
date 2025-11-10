## Rossmann Store Sales

Forecast sales using store, promotion, and competitor data

![](https://www.kaggle.com/competitions/4594/images/header)

## Rossmann Store Sales

### Description

Rossmann operates over 3,000 drug stores in 7 European countries. Currently, Rossmann store managers are tasked with predicting their daily sales for up to six weeks in advance. Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality. With thousands of individual managers predicting sales based on their unique circumstances, the accuracy of results can be quite varied.

In their first Kaggle competition, Rossmann is challenging you to predict 6 weeks of daily sales for 1,115 stores located across Germany. Reliable sales forecasts enable store managers to create effective staff schedules that increase productivity and motivation. By helping Rossmann create a robust prediction model, you will help store managers stay focused on what’s most important to them: their customers and their teams! 

![](https://storage.googleapis.com/kaggle-media/competitions/kaggle/4594/media/rossmann_banner2.png)

### Evaluation

Submissions are evaluated on the Root Mean Square Percentage Error (RMSPE). The RMSPE is calculated as

$$
\mathrm{RMSPE}
=\sqrt{\frac{1}{n}\sum_{i=1}^{n}\left(\frac{y_i-\hat{y}_i}{y_i}\right)^2}
$$

where y_i denotes the sales of a single store on a single day and yhat_i denotes the corresponding prediction. Any day and store with 0 sales is ignored in scoring.

## Submission File

The file should contain a header and have the following format:

```
Id,Sales
1,0
2,0
3,0
etc.
```

### Prizes

- 1st place - $15,000
- 2nd place - $10,000
- 3rd place - $5,000
- In addition, a single $5,000 reward will go to the team whose methodology is implemented by Rossmann. This award may be given to a team at any position on the leaderboard.

Rossmann is interested in hiring top Kagglers from this competition. If you're interested in a position with Rossmann, you may append "JOB" next to your team name for consideration.

### Timeline

- **December 7, 2015** - First Submission deadline. Your team must make its first submission by this deadline.
- **December 7, 2015** - Team Merger deadline. This is the last day you may merge with another team.
- **December 14, 2015** - Final submission deadline

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The organizers reserve the right to update the contest timeline if they deem it necessary.

### Citation

FlorianKnauer and Will Cukierski. Rossmann Store Sales. https://kaggle.com/competitions/rossmann-store-sales, 2015. Kaggle.
