	% Classify texts according to the language in which they are written.
	% The texts are input interactively. To quit the program, enter "quit".
	
	% *** First part: Initialize; then read the trigram counts for the various languages
	
    clear
	languages = {'pt', 'es', 'fr', 'en'};			% Languages that can be recognized
	basefilename = '_trigram_count_filtered.tsv';	% Fixed part of the trigram counts' filenames

	% Load the trigram counts

	fprintf('\nLoading trigram counts: ')

	for languageindex = 1:numel(languages)						% Loop on the languages, for reading the trigram counts
		temp_count = containers.Map('KeyType','char','ValueType','double');		% Associative array for temporarily storing trigram counts
		language = languages{languageindex};
		
		fprintf('%s... ', language)							% Give feedback to the user, because this may take some time

		total = 0;											% Temporary variable that will contain the total counts for this language
		filename = strcat(language, basefilename);			% Build the name of the counts file to read
		fileid = fopen(filename, 'r', 'n', 'UTF-8');
		line = fgetl(fileid);								% Read the first line of the file
		while ischar(line)									% Loop on the successive lines of the file
			trigram = line(1:3);							% Trigram
			trigram_count = str2double(line(4:end));		% Trigram's count in this language
			temp_count(trigram) = trigram_count;			% Store the trigram's count
			total = total + trigram_count;					% Update the total counts
			line = fgetl(fileid);							% Read the next line
		end
		fclose(fileid);

		counts{languageindex} = temp_count;								% Store the counts for this language
		total_counts(languageindex) = total;							% Store the total counts for this language
	end



	% *** Second part: This is a loop which repeatedly asks for text and then classifies it

	fprintf('\n\nType a line of text to be classified. To quit, type "quit"\n\n')
	
	while 1			% Main loop: Read the input text and then classify it
		
		text = input('Text: ', 's');		% Read a line of input text
		if strcmp(text, 'quit')				% Check whether it is the 'quit' command
			break							% If it is the 'quit' command, exit the main loop
		end
		text = lower(text);					% Change all text to lower-case

		for languageindex = 1:numel(languages)          % Loop through the available languages
				
	% SECTION 1		
            % Additive smoothing (Laplce Smoothing) requires that we add 1
            % to all triagram combinations, since the set of characters
            % used is of length 60, the sum of ones of all all combinations
            % equals 60^3 (used update the total_count for each language)
            
            v_total_counts(languageindex)=0;
            v_total_counts(languageindex)=total_counts(languageindex)+60^3;
            
            clear v_trigramcount;
            
			
			for trigramindex = 1:numel(text)-2;					% Loop through all the trigrams of the input text
				
				trigram = text(trigramindex:trigramindex+2);	% Contains the trigram that is to be processed in this iteration
				
				if isKey(counts{languageindex}, trigram)			% Check whether the trigram exists in the counts list
					trigramcount = counts{languageindex}(trigram);		% Contains the number of times that the current trigram occurred in the training data for the current language
				else
					trigramcount = 0;
				end

            
				
           % SECTION 2   
                % We also update de trigram count, by adding 1, as a result
                % of using additive smoothing
                
               
                v_trigramcount(trigramindex) = log(trigramcount + 1);
                
				
			end		% The loop on trigrams ends here
                    
	% SECTION 3 
            scores(languageindex) = sum(v_trigramcount)-(length(v_trigramcount)*log(v_total_counts(languageindex)));
            
		end			% The loop on languages ends here
		
		% Output the results
		
		fprintf('\nScores:\n')
		
		for languageindex = 1:numel(languages)
			fprintf('%s: %10.8g\n', languages{languageindex}, scores(languageindex))
		end

		[~, sorted_indexes] = sort(scores, 'descend');		% Sort the scores in descending order, and obtain the corresponding language indexes

		fprintf('\nRecognized language: %s,   classification margin = %10.8g\n\n',languages{sorted_indexes(1)}, scores(sorted_indexes(1)) - scores(sorted_indexes(2)))/log(4);
	end

	fprintf('Finished\n\n')
