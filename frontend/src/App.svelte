<script>
    let file;
    let result = '';

    const uploadImage = async () => {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('http://localhost:5000/analyze', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        result = JSON.stringify(data.prediction);
    };
</script>

<main>
    <h1>Reticulocyte Detection</h1>
    <input type="file" bind:this={file} />
    <button on:click={uploadImage}>Upload</button>
    {#if result}
        <p>Result: {result}</p>
    {/if}
</main>
