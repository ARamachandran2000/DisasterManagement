import React from 'react';
import { Route, Link, BrowserRouter as Router } from 'react-router-dom';
import Headlines from './headlines';
import Categories from './categories';

class Header extends React.Component{
    render(){
        return (
            <React.Fragment>
                <div className="head">
                    <div className="headerobjectswrapper">
                        <header>News Updates</header>
                    </div>
                    {/* <div class="subhead">Headlines</div> */}
                </div>

                <div className="cat_list">
                <Router>
                    <div className="router_wrap">
                    <ul className="navList">
                        <li><Link to="/categories/earthquake damage">Earthquake</Link></li>
                        <li><Link to="/categories/fire in forest">Forest Fire</Link></li>
                    </ul>
                    <Route exact path="/" component={Headlines} />
                    <Route path="/categories/:name" component={Categories} />
                    </div>
                </Router>
                </div>
            </React.Fragment>
        );
    }
}

export default Header;