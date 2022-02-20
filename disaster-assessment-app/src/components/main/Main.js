import "./Main.css";
import hello from "../../assets/ic_launcher.png"
import blue from "../../assets/blue.png"
import red from "../../assets/red.png"
import green from "../../assets/green.png"
import { useState,useEffect } from "react";
import Sidebar from "../sidebar/Sidebar";
import { e_array } from "../sidebar/Sidebar";


import Chart from "../charts/Chart";
const Main=(props)=>{
    console.log(props);
    return(
        <main>
            <div className="main__container">
                <div className="main__title">
                    
                    <div className="main__greeting">
                        <h1 style={{fontSize: '34px'}}>Disaster Assessment</h1>
                        
                    </div>
                </div>
                

                <div className="charts">
                   
                    <div className="charts__right">
                        <div className="charts__left__title">
                            <div>
                                <h1 style={{fontSize: '26px'}}>Output Image</h1>
                            </div>
                        </div>
                        <div className="charts__right__cards">
                        <img src={props.image==null?hello:props.image} style={{width:'480px', height:'480px', padding: '20px'}} id="img"></img>
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
                    <div className="card" >
                        <i className="fa fa-users fa-2x text-lightblue"></i>
                        <div className="card_inner">
                            <p className="text-primary-p">Earthquake Magnitude</p>
                            <span className="font-bold text-title">3.3</span>
                        </div>
                    </div>
                </div>
                </div>
            </div>
        </main>
    )
}

export default Main;