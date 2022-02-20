import "./Main.css";
import hello from "../../assets/loading_1.gif"
import blue from "../../assets/blue.png"
import red from "../../assets/red.png"
import green from "../../assets/green.png"
import { useState,useEffect } from "react";
import Sidebar from "../sidebar/Sidebar";
import { e_array } from "../sidebar/Sidebar";
import { BrowserRouter as Router, Route, Link, Switch } 
       from "react-router-dom";


import Chart from "../charts/Chart";
const Main=(props)=>{
    console.log(props);
    return(
        <main>
            <div className="main__container">
                <div className="main__title">
                    
                    <div className="main__greeting">
                        <h1 style={{fontSize: '34px', marginLeft:'115px'}}>Disaster Assessment</h1>
                        
                    </div>
                </div>
                

                <div className="charts">
                   
                    <div className="charts__right">
                        <div className="charts__left__title">
                            <div>
                                <h1 style={{fontSize: '26px', marginLeft:'150px'}}>Output Image</h1>
                            </div>
                        </div>
                        <div className="charts__right__cards">
                        <img src={props.image==null?hello:props.image} style={{width:'350px', height:'350px', padding: '40px', paddingLeft:'55px'}} id="img"></img>
                        </div>
                        
                        <div className="charts__right__colors">
                        <img src={red} sizes={50}></img>
                        <span>Severe Damage</span>
                        <img src={blue}></img>
                        <span>Moderate Damage</span>
                        <img src={green}></img>
                        <span>No Damage</span>
                        </div>
                    </div>

                    <div className="charts_left">
                    <div className="card" >
                        <i className="fa fa-users fa-2x text-lightblue"></i>
                        <div className="card_inner">
                            <p className="text-primary-p">Area Damage (%)</p>
                            <span className="font-bold text-title">{props.areaDamage} %</span>
                        </div>
                    </div>
                    <br/>
                    
                    <form action="http://localhost:3005/categories/earthquake%20damage" target="_blank">
                        <input type="submit" value="Check Live Updates" style={{marginLeft:'150px', backgroundColor:'lightblue', padding: '10px', borderRadius: '12px', cursor: 'pointer', fontWeight: 'bold', padding: '10px 15px', textAlign: 'center', transition: '100ms', maxWidth:'180px', boxSizing:'border-box', border: '0', fontSize:'16px', userSelect:'none', WebkitUserSelect:'none', touchAction:'manipulation',marginTop:'15px'}}/>
                    </form>
                    
                </div>
                </div>
            </div>
        </main>
    )
}

export default Main;