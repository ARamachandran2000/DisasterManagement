import React from 'react';
import axios from 'axios';

class Categories extends React.Component{
    state = {
        headlinesNews: [],
        isLoading: true,
        errors: null,
        urlParams: this.props.match.params
    };

    getUsers(catName) {
        // I have used axios instead of Fetch
        // axios.get('https://newsapi.org/v2/top-headlines',{
        //     params: {country: 'IN', apiKey: 'b1f8f098191a4ab5ae0581b74565cbf3'}
      // })
        axios.get('https://gnews.io/api/v4/search', {
          params:{lang:'en', q: catName, token: '7a3abe97da7c5c17ed09ba31a208a6c8'}
        })
          // mapping the API endpoints to props
          .then(response =>
            response.data.articles.map(news => ({
              title: `${news.title}`,
              description: `${news.description}`,
              newsurl: `${news.url}`,
              url: `${news.image}`,
              content: `${news.content}`
            }))
          )
          // Updating loading state
          .then(headlinesNews => {
            this.setState({
              headlinesNews,
              isLoading: false
            });
          })
          // using the `.catch()` method since axios is promise-based
          .catch(error => this.setState({ error, isLoading: false }));
    }

    componentWillReceiveProps(nextProps) {
        if (nextProps.match.params !== this.props.match.params) {
          const currentName = nextProps.match.params
          this.getUsers(currentName);
          this.setState({
            urlParams: currentName
          })
        }
      }

    componentDidMount() {
        const { name } = this.props.match.params;
        this.getUsers(name);
    }

    render(){
        const { isLoading, headlinesNews } = this.state;
        return (
            <React.Fragment>
            <div className="subhead"><h2>{this.state.urlParams.name}</h2></div>
            <div className="news_content">
                {!isLoading ? (
                headlinesNews.map(news => {
                    const { title, description, author, newsurl, url, content } = news;
                    return (
                    <div className="collumn">
                        <div className="head">
                            <span className="headline hl3">
                                {title}
                            </span>
                            <p>
                                <span className="headline hl4">
                                    {author}
                                </span>
                            </p>
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

export default Categories;