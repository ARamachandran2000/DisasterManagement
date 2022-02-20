
import axios from 'axios';
 
import React,{Component} from 'react';

class App extends Component {

    // constructor() {
    //   super();
    //   this.state = {
    //     model: ''
    //   };
    // }

    state = {
      selectedFile: null,
      model: null
    };
    
    onFileChange = event => {
      this.setState({ selectedFile: event.target.files[0] });
    };
    
    onRadioChange = event => {
      this.setState({model: event.target.value})
    }

    onFileUpload = () => {
      const formData = new FormData();
      formData.append(
        "myFile",
        this.state.selectedFile,
        this.state.selectedFile.name
      );

      formData.append(
        "selectedModel",
        this.state.model
      );

      axios.post("http://localhost:5000/test", formData)
      .then(response => {
        this.props.changeImage(response.data.url);
        this.props.changeDamageVal(response.data.areaDamage)
      } )
      .catch(error => {
        console.log("H");
        console.log(error)
      })
    };
    
    fileData = () => {
      if (this.state.selectedFile) {
        return (
          <div>
            {this.state.selectedFile && <ImageThumb image={this.state.selectedFile} />}
          </div>
        );
      } else {
        return (
          <div>
            <br />
            <h4>Choose before Pressing the Upload button</h4>
          </div>
        );
      }
    };
    
    render() {
      return (
        <div>
            <div className="sidebar__title" style={{padding: '20px', paddingTop: '130px'}}>
                  <div className="sidebar__image">
                      <h3 style={{fontSize: '26px'}}>Select a Satellite Image</h3>
                  </div>
            
          {this.fileData()}
          </div>
                <input type="file" onChange={this.onFileChange} style={{paddingLeft: '20px', alignContent: "center", justifyContent: "center", display:"flex"}} />
                <ul>
                  <li>
                    <label>
                      <input
                        type="radio"
                        value="earthquake"
                        checked={this.state.model === "earthquake"}
                        onChange={this.onRadioChange}
                      />
                      <span>Earthquake Damage</span>
                    </label>
                  </li>
                  <li>
                    <label>
                      <input
                        type="radio"
                        value="fire"
                        checked={this.state.model === "fire"}
                        onChange={this.onRadioChange}
                      />
                      <span>Fire Detection</span>
                    </label>
                  </li>
                </ul>


                <div className="App" style={{paddingLeft: '20px', alignContent: "center", justifyContent: "center", display:"flex"}}>
                      <button type="button" id = "assess" onClick={this.onFileUpload} style={{backgroundColor:"#fff000", padding: '10px', borderRadius: '12px', cursor: 'pointer', fontWeight: 'bold', padding: '10px 15px', textAlign: 'center', transition: '100ms', maxWidth:'180px', boxSizing:'border-box', border: '0', fontSize:'16px', userSelect:'none', WebkitUserSelect:'none', touchAction:'manipulation',marginTop:'15px'}}> ASSESS </button>
               </div> 
          
        </div>
        
      );
    }
    
  }
 
  const ImageThumb = ({ image }) => {
    return <img src={URL.createObjectURL(image)} alt={image.name} style={{width:'480px', height:'320px', padding: '20px'}} />;
};
  export default App;


