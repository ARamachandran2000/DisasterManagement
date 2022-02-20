import React from 'react';
import axios from 'axios';

class Headlines extends React.Component{

    constructor(props){
        super(props)
        this.state = {
            headlinesNews: [],
            isLoading: true,
            errors: null
        };
    }

    getUsers(catags) {
        // We're using axios instead of Fetch
        axios.get('https://newsapi.org/v2/top-headlines',{
            params: {country: catags, apiKey: 'b1f8f098191a4ab5ae0581b74565cbf3'}
        })
          // Once we get a response, we'll map the API endpoints to our props
          .then(response =>
            response.data.articles.map(news => ({
              title: `${news.title}`,
              description: `${news.description}`,
              author: `${news.author}`,
              newsurl: `${news.url}`,
              url: `${news.urlToImage}`,
              content: `${news.content}`
            }))
          )
          // Let's make sure to change the loading state to display the data
          .then(headlinesNews => {
            this.setState({
              headlinesNews,
              isLoading: false
            });
          })
          // We can still use the `.catch()` method since axios is promise-based
          .catch(error => this.setState({ error, isLoading: false }));
    }

    componentDidMount() {
        this.getUsers('us')
    }

    render(){
        const { isLoading, headlinesNews } = this.state;
        return (
            <React.Fragment>
            <div className="subhead"><h2>Headlines</h2></div>
            <div>
                {!isLoading ? (
                headlinesNews.map(news => {
                    const { title, description, author, newsurl, url, content } = news;
                    return (
                    <div className="collumn" key={title}>
                        <div className="head">
                            <span className="headline hl3">
                                {title}
                            </span>
                            {/* <p>
                                <span className="headline hl4">
                                    {author}
                                </span>
                            </p> */}
                            <figure className="figure">
								<img className="media" src={url} alt="" />
						    </figure>
                            <p>
                                {description}<br />
                                {content}
                            </p>
                            <a href={newsurl} target="_blank" rel="noopener noreferrer">Read full news</a>
                        </div>
                    </div>
                    );
                })
                ) : (
                <p>Loading...</p>
                )}
            </div>
            </React.Fragment>
        );
    }
}

export default Headlines;