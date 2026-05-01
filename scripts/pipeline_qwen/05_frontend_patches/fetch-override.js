// Patch util.getFile's cache mode to 'default' (respects Cache-Control) instead                                                
// of 'force-cache' (holds stale errors forever). Also strip credentials for                                                    
// cross-origin HF requests so browser cookies don't trigger 401s.                                                              
(function () {                                                                                                                  
    function waitForUtil() {                                                                                                    
        if (!window.util || !window.util.getFile) {                                                                             
            setTimeout(waitForUtil, 50);
            return;                                                                                                             
        }       
        const original = window.util.getFile;                                                                                   
        // Monkey-patch by replacing the fetch() call's options via a wrapper on window.fetch
        // for the huggingface.co origin only. Leaves all other fetches untouched.                                              
        const nativeFetch = window.fetch.bind(window);                                                                          
        window.fetch = function (input, init) {                                                                                 
            const url = (typeof input === 'string') ? input : (input && input.url) || '';                                       
            if (url.includes('huggingface.co') || url.includes('cloudfront.net')) {
                const patched = Object.assign({}, init || {}, {                                                                 
                    cache: 'default',                                                                                           
                    credentials: 'omit',                                                                                        
                    mode: 'cors',                                                                                               
                });                                                                                                             
                return nativeFetch(input, patched);
            }                                                                                                                   
            return nativeFetch(input, init);
        };
        console.log('[fetch-override] wrapping fetch for HF + CloudFront');
    }                                                                                                                           
    waitForUtil();
})();