export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {
      // Add specific options for better browser compatibility
      overrideBrowserslist: [
        '>0.2%',
        'not dead',
        'not op_mini all'
      ]
    },
  },
}
