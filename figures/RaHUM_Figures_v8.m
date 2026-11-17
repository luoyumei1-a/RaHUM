function RaHUM_Figures_v8(outDir, figureToDraw)
% Article-ordered MATLAB plotting source for RaHUM v8.
% Uses actual recorded TRAIN losses and strictly restored adapter diagnostics.
% No benchmark accuracy, generated-claim rate, confusion label or uncertainty
% interval is inferred from these engineering observations.
% Native checkpoint SHA256: f39fc35d0d766c86485f3c630448e63554f7dac5b75a969da1ee7a2b6b2bfa76
% Real feature-cache SHA256: 01f2e9c765fb4efef8f4df8f77e70adfb6bf4c73f4cf5683139d7615520720a0
% Exact source data and execution receipts accompany the code release.
% Usage: RaHUM_Figures_v8('figures','all')
% Article order: Fig2, Fig3, Fig10, Fig11, Fig12. Fig10/11 are separate square
% figures placed next to each other in one manuscript column, without subfigure.
if nargin < 1 || isempty(outDir), outDir = 'figures'; end
if nargin < 2 || isempty(figureToDraw), figureToDraw = 'all'; end
if ~exist(outDir,'dir'), mkdir(outDir); end
fontName = 'Times New Roman';
colors.radar = [211 72 54]/255;
colors.reference = [48 100 141]/255;
colors.light = [154 213 221]/255;
colors.grid = [.85 .87 .89];
% Full-precision observations. Populations are 42 geometrically supported
% groups from one three-frame recorded window, not 42 independent scenes.
utility = [0.657035231590271 0.60460424423217773 0.65224099159240723 0.62695789337158203 0.5911794900894165 0.61459189653396606 0.61647850275039673 0.66060584783554077 0.71605151891708374 0.60753971338272095 0.59205329418182373 0.61417770385742188 0.61618322134017944 0.64278149604797363 0.60123360157012939 0.50324314832687378 0.60993695259094238 0.59689199924468994 0.58757388591766357 0.55488240718841553 0.6355927586555481 0.55877900123596191 0.43875351548194885 0.61032354831695557 0.63731253147125244 0.63636744022369385 0.6143336296081543 0.64422935247421265 0.57884025573730469 0.63810467720031738 0.53983443975448608 0.57521259784698486 0.59119391441345215 0.57116425037384033 0.545482337474823 0.52939081192016602 0.57928425073623657 0.59589040279388428 0.61243796348571777 0.56753289699554443 0.63044387102127075 0.58370250463485718];
maxAttention = [0.19507764279842377 0.26739901304244995 0.19876320660114288 0.19930239021778107 0.24918480217456818 0.22649016976356506 0.22857329249382019 0.19498704373836517 0.21938714385032654 0.23033025860786438 0.25868478417396545 0.22008223831653595 0.23841746151447296 0.18220977485179901 0.19721364974975586 0.27843949198722839 0.19707103073596954 0.18221944570541382 0.22676578164100647 0.22731034457683563 0.20092985033988953 0.19469305872917175 0.20863601565361023 0.2198871523141861 0.2654607892036438 0.22156044840812683 0.22258728742599487 0.19361002743244171 0.19619525969028473 0.21639068424701691 0.21736578643321991 0.20596463978290558 0.25287041068077087 0.26966843008995056 0.20777355134487152 0.21668067574501038 0.24442248046398163 0.20658816397190094 0.18298013508319855 0.22958871722221375 0.2011561244726181 0.19530549645423889];
frameAttention = [0.19958017766475677 0.15944790840148926 0.15182366967201233 0.16457535326480865 0.093366928398609161 0.088767118752002716 0.084982916712760925 0.057455915957689285; 0.18348859250545502 0.19900885224342346 0.14593519270420074 0.13751634955406189 0.10065975785255432 0.096042878925800323 0.070241622626781464 0.067106775939464569; 0.18138228356838226 0.1819135993719101 0.15494643151760101 0.14451657235622406 0.10705883800983429 0.096928238868713379 0.06682993471622467 0.066424086689949036];
trainCE = [2.6411845684051514 2.4695336818695068; 2.6411845684051514 2.3584704399108887];
localK = [1 2 4 8 12 16 24 32];
entropy = [0 0.67703545377368024 1.3556306191853114 1.9824524465061368 2.3108263867241994 2.4958344527653287 2.6687286297480264 2.7339890457334972];
forwardMs = [19.888990999788803 17.479278999417147 17.227013000592706 20.206643000165059 26.798175999829255 30.789577000177815 34.521116999712831 40.491740000106802];
assert(numel(utility)==42 && numel(maxAttention)==42);
assert(max(abs(sum(frameAttention,2)-1)) < 1e-5);
assert(all(trainCE(:)>0));
order = {'fig02_claim_types','fig03_weather','fig10_memory_entropy', ...
    'fig11_adapter_time','fig12_reference_attention'};
if strcmpi(figureToDraw,'all')
    chosen = order;
elseif isnumeric(figureToDraw)
    chosen = order(figureToDraw);
else
    chosen = cellstr(string(figureToDraw));
end
for ii=1:numel(chosen)
    stem=chosen{ii};
    switch stem
        case 'fig02_claim_types'
            f = canvas(16.2,7.0,stem);
            ax=axes(f,'Position',[.105 .235 .36 .70]);
            histogram(ax,utility,.35:.05:.80,'FaceColor',colors.radar, ...
                'EdgeColor','w','FaceAlpha',1,'LineWidth',.8);
            xlabel(ax,'Radar utility'); ylabel(ax,'Groups');
            xlim(ax,[.35 .80]); xticks(ax,[.4 .5 .6 .7 .8]);
            ylim(ax,[0 22]); yticks(ax,[0 5 10 15 20]);
            styleAxes(ax,18);
            ax=axes(f,'Position',[.605 .235 .36 .70]);
            histogram(ax,maxAttention,.15:.025:.325,'FaceColor',colors.reference, ...
                'EdgeColor','w','FaceAlpha',1,'LineWidth',.8);
            xlabel(ax,'Maximum attention'); ylabel(ax,'Groups');
            xlim(ax,[.15 .325]); xticks(ax,[.15 .20 .25 .30]);
            ylim(ax,[0 22]); yticks(ax,[0 5 10 15 20]);
            styleAxes(ax,18);
        case 'fig03_weather'
            f=canvas(13.2,6.8,stem);
            ax=axes(f,'Position',[.16 .22 .79 .73]); hold(ax,'on');
            plot(ax,1:2,trainCE(1,:),'-s','Color',colors.reference, ...
                'MarkerFaceColor',colors.reference,'MarkerSize',7,'LineWidth',1.65);
            plot(ax,1:2,trainCE(2,:),'-o','Color',colors.radar, ...
                'MarkerFaceColor',colors.radar,'MarkerSize',7,'LineWidth',1.65);
            xlabel(ax,'Optimizer update'); ylabel(ax,'TRAIN token CE');
            xlim(ax,[.92 2.08]); xticks(ax,1:2); ylim(ax,[2.28 2.72]);
            legend(ax,{'Projector control','RaHUM'},'Location','southwest', ...
                'FontName',fontName,'FontSize',16,'Box','off');
            styleAxes(ax,18);
        case 'fig10_memory_entropy'
            f=canvas(12.0,11.4,stem);
            ax=axes(f,'Position',[.20 .18 .75 .77]);
            plot(ax,1:numel(localK),entropy,'-o','Color',colors.radar, ...
                'MarkerFaceColor',colors.radar,'MarkerSize',8,'LineWidth',1.65);
            xticks(ax,1:numel(localK)); xticklabels(ax,string(localK));
            xlim(ax,[.7 numel(localK)+.3]); ylim(ax,[0 3]);
            yticks(ax,0:1:3);
            xlabel(ax,'Local memory K'); ylabel(ax,'Attention entropy (nats)');
            styleAxes(ax,22);
        case 'fig11_adapter_time'
            f=canvas(12.0,11.4,stem);
            ax=axes(f,'Position',[.20 .18 .75 .77]);
            bar(ax,1:numel(localK),forwardMs,.67,'FaceColor',colors.reference, ...
                'EdgeColor','none');
            xticks(ax,1:numel(localK)); xticklabels(ax,string(localK));
            xlim(ax,[.4 numel(localK)+.6]); ylim(ax,[0 45]); yticks(ax,0:15:45);
            xlabel(ax,'Local memory K'); ylabel(ax,'Adapter time (ms)');
            styleAxes(ax,22);
        case 'fig12_reference_attention'
            f=canvas(13.2,7.4,stem);
            ax=axes(f,'Position',[.14 .23 .71 .70]);
            imagesc(ax,frameAttention,[0 .25]);
            colormap(ax,parula(256));
            xticks(ax,1:8); yticks(ax,1:3);
            xlabel(ax,'Local reference rank'); ylabel(ax,'Frame');
            c=colorbar(ax); c.Label.String='Mean attention';
            c.FontName=fontName; c.FontSize=16;
            for row=1:3
                for col=1:8
                    v=frameAttention(row,col);
                    if v>.17, textColor='k'; else, textColor='w'; end
                    text(ax,col,row,sprintf('%.2f',v),'HorizontalAlignment','center', ...
                        'FontName',fontName,'FontSize',14,'Color',textColor);
                end
            end
            styleAxes(ax,18); ax.YGrid='off'; ax.XGrid='off';
        otherwise
            error('Unknown figure: %s',stem);
    end
    set(findall(f,'-property','FontName'),'FontName',fontName);
    exportgraphics(f,fullfile(outDir,[stem '.pdf']),'ContentType','vector','BackgroundColor','white');
    exportgraphics(f,fullfile(outDir,[stem '.png']),'Resolution',220,'BackgroundColor','white');
    savefig(f,fullfile(outDir,[stem '.fig']));
    fprintf('EXPORTED %s\n',stem);
    close(f);
end
fid=fopen(fullfile(outDir,'MATLAB_execution.txt'),'w');
fprintf(fid,'MATLAB %s\nRelease %s\n',version,version('-release'));
fprintf(fid,'Source populations: 42 supported radar groups, 3 frames, one feature window.\n');
fprintf(fid,'Losses: 2 TRAIN updates. Local K timing: five repeats after warm-up.\n');
fclose(fid);
    function f=canvas(w,h,name)
        f=figure('Name',name,'Color','white','Units','centimeters', ...
            'Position',[2 2 w h],'PaperUnits','centimeters', ...
            'PaperPosition',[0 0 w h],'PaperSize',[w h],'Visible','off');
    end
    function styleAxes(ax,sz)
        set(ax,'FontName',fontName,'FontSize',sz,'LineWidth',1.25, ...
            'Box','on','TickDir','out','Layer','top','GridColor',colors.grid, ...
            'GridAlpha',.65,'YGrid','on','XGrid','off');
        ax.XLabel.FontSize=sz+2; ax.YLabel.FontSize=sz+2;
    end
end
