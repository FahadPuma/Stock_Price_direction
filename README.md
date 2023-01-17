## **Stock_Price_direction**
To generate recommendations for a stock price movement based upon technical, Regression and Sentiment analysis

### **Problem Statement:**
Retail investors who are subscribing to different brokerage and investments tips portals find it difficult to pick the stocks to BUY/ SELL real-time without any latency and are often taking decisions based on emotions or past performance of a specific scrip. This may somehow not predict the precise current or future direction of the stocks. 

With this proposed prediction model, the objective is to provide an investor a real-time scores of different technical and sentiment analysis of a specific group of stocks based on historical as well current market sentiments.

![Picture 1](https://user-images.githubusercontent.com/87992010/212847237-22021147-4fd1-4b3d-8e68-e8cafcfc262c.png)


### **Solution:**
The proposed solution takes both historical as well current market situation of a specific stock into account. 
- First step is - Different technical indicators and their respective scores are fed into a logistic regression model to predict whether a stock’s price will go up or down in the next time frame.
- Second step is - Designing a logistic regression model based upon these technical indicators and their respective scores to predict the direction(up/ down) of a stock in the next time frame
- Third step is - To do web scrapping of various business news portals and social media to collect all the recent news, recommendations and discussions related to a specific share. NLP is performed on this collected unstructured data to perform Sentiment Analysis. 
Based upon these 3 scores, the final prediction is provided to the investor for the stock to whether or not BUY/ SELL it.
### <ins>**Solution Architecture:**</ins>
![Picture 3](https://user-images.githubusercontent.com/87992010/212849451-1752b678-50e0-413b-b8dd-7c079d70273b.png)
Below are the scope of the solution:
1.	Collection of historical data for the specified list of stocks
2.	Identifying and choosing technical indicators from “Leading”, “Lagging”, “Trend”, “Volatility” categories
3.	Designing an algorithm based upon price action and technical indicators' combination to to predict the direction of the stock in the next timeframe
4.	Designing a logistic regression model based upon these technical indicators and their respective scores to predict the direction(up/ down) of a stock in the next timeframe
5.	Web scrapping of various business news portals, investment advisor sites, social media to collect unstructured data for a specific stock
6.	Performing a Sentiment Analysis using NLP on the collected unstructured data to present “Positive” or “Negative” sentiments for a stock
7.	Comparison of the 3 models and their respective scores to arrive at the final prediction model of a stock price movement in the future
8.	Decision to perform BUY/ SELL on a stock

### <ins>**Final Recommendation**</ins>
![Picture 4](https://user-images.githubusercontent.com/87992010/212849863-7a2297d1-2824-4378-81b9-937515b8f9a7.png)

<u>Based upon the above table, user can make a decision about the purchase or sell of a stock in the future. As observed from the above table, it provides a comprehensive analysis and the corresponding scores taking all the aspects and factors which may cause the volatility and the movement of the stock. This also enables user to plan his short term and long term investments using the scores based upon the different timeframes (1m, 5m ,1h,1d etc. for short term or 1week, 1 month etc. for long term).</u>
